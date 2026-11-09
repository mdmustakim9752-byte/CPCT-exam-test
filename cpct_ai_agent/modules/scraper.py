"""
Module 1: CPCT Auto-Downloader & Unzipper (Web Scraper)
Automates fetching previous year question papers from the official MP CPCT portal.
"""

import os
import re
import ssl
import sys
import zipfile
import urllib.parse
from pathlib import Path
from typing import List, Dict, Optional, Any
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import urllib3

# Suppress insecure HTTPS request warnings for government portals with self-signed certs
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Ensure project root is in sys.path for direct execution
current_file = Path(__file__).resolve()
agent_root = current_file.parent.parent
workspace_root = agent_root.parent
for p in [str(workspace_root), str(agent_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from cpct_ai_agent.config import (
        CPCT_PORTAL_URL,
        REQUEST_HEADERS,
        REQUEST_TIMEOUT,
        RAW_ZIPS_DIR,
        EXTRACTED_PDFS_DIR,
    )
except ImportError:
    from config import (
        CPCT_PORTAL_URL,
        REQUEST_HEADERS,
        REQUEST_TIMEOUT,
        RAW_ZIPS_DIR,
        EXTRACTED_PDFS_DIR,
    )



class CPCTScraper:
    """
    Automated scraper for MP CPCT (Computer Proficiency Certification Test)
    previous year question papers and shift-wise PDFs.
    """

    def __init__(
        self,
        portal_url: str = CPCT_PORTAL_URL,
        raw_zips_dir: Path = RAW_ZIPS_DIR,
        extracted_pdfs_dir: Path = EXTRACTED_PDFS_DIR,
    ):
        self.portal_url = portal_url
        self.raw_zips_dir = Path(raw_zips_dir)
        self.extracted_pdfs_dir = Path(extracted_pdfs_dir)
        self.raw_zips_dir.mkdir(parents=True, exist_ok=True)
        self.extracted_pdfs_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(REQUEST_HEADERS)

    def fetch_portal_html(self) -> str:
        """Fetch the HTML content of the CPCT Previous Question Papers page."""
        try:
            resp = self.session.get(
                self.portal_url,
                verify=False,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding or "utf-8"
            return resp.text
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to fetch CPCT portal page: {e}") from e

    def get_available_papers(self) -> List[Dict[str, str]]:
        """
        Parse the portal table and extract all available question paper cycles.
        Returns a list of dicts:
            [{
                'exam_date': '5th Sep 2026',
                'title': 'QP PDF 5th Sep 2026',
                'qp_url': 'https://.../5th_Sep_2026.zip',
                'notification_url': 'https://.../Notification.zip'
            }, ...]
        """
        html = self.fetch_portal_html()
        soup = BeautifulSoup(html, "html.parser")
        papers = []

        # Find all table rows
        rows = soup.find_all("tr")
        for row in rows:
            tds = row.find_all("td")
            if not tds or len(tds) < 2:
                continue

            row_text = row.get_text(" ", strip=True)
            if "QP PDF" not in row_text and ".zip" not in row_text.lower() and ".pdf" not in row_text.lower():
                continue

            exam_date = tds[0].get_text(strip=True)
            if not exam_date or "Exam Date" in exam_date:
                continue

            # Extract Question Paper link from second column
            qp_td = tds[1]
            qp_anchor = qp_td.find("a", href=True)
            if not qp_anchor:
                continue

            qp_relative_url = qp_anchor["href"].strip()
            qp_url = urllib.parse.urljoin(self.portal_url, qp_relative_url)
            qp_title = qp_anchor.get_text(strip=True) or f"QP {exam_date}"

            # Optional notification link
            notification_url = ""
            if len(tds) > 2:
                notif_anchor = tds[2].find("a", href=True)
                if notif_anchor:
                    notification_url = urllib.parse.urljoin(self.portal_url, notif_anchor["href"].strip())

            papers.append({
                "exam_date": exam_date,
                "title": qp_title,
                "qp_url": qp_url,
                "notification_url": notification_url,
            })

        return papers

    def download_paper(self, paper: Dict[str, str], force: bool = False) -> Path:
        """
        Download a single question paper zip archive or PDF.
        """
        url = paper["qp_url"]
        exam_date = paper["exam_date"]

        # Normalize filename
        parsed_url = urllib.parse.urlparse(url)
        original_filename = os.path.basename(parsed_url.path) or "paper.zip"
        # Sanitize filename
        safe_date = re.sub(r"[^\w\-]", "_", exam_date)
        if not original_filename.lower().endswith((".zip", ".pdf")):
            original_filename += ".zip"

        dest_filename = f"{safe_date}_{original_filename}"
        dest_path = self.raw_zips_dir / dest_filename

        if dest_path.exists() and not force and dest_path.stat().st_size > 0:
            print(f"[*] Already downloaded: {dest_path.name}")
            return dest_path

        print(f"[+] Downloading {paper['title']} from {url}...")
        try:
            resp = self.session.get(url, stream=True, verify=False, timeout=60)
            resp.raise_for_status()

            total_size = int(resp.headers.get("content-length", 0))
            with open(dest_path, "wb") as f, tqdm(
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
                desc=dest_path.name,
                leave=False,
            ) as progress:
                for chunk in resp.iter_content(chunk_size=16384):
                    if chunk:
                        size = f.write(chunk)
                        progress.update(size)

            print(f"[OK] Downloaded: {dest_path.name} ({dest_path.stat().st_size / 1024:.1f} KB)")
            return dest_path
        except Exception as e:
            if dest_path.exists():
                dest_path.unlink(missing_ok=True)
            raise RuntimeError(f"Failed to download {url}: {e}") from e

    def extract_zip(self, zip_path: Path, exam_date: str = "") -> List[Path]:
        """
        Unzip a downloaded CPCT question paper archive and extract shift-wise PDFs.
        Flattens directory structure into:
            extracted_pdfs/<exam_date>/<pdf_file>
        """
        zip_path = Path(zip_path)
        if not zip_path.exists():
            raise FileNotFoundError(f"Zip file not found: {zip_path}")

        # If it's already a PDF, copy directly
        if zip_path.suffix.lower() == ".pdf":
            target_folder = self.extracted_pdfs_dir / (re.sub(r"[^\w\-]", "_", exam_date) or "misc")
            target_folder.mkdir(parents=True, exist_ok=True)
            target_pdf = target_folder / zip_path.name
            target_pdf.write_bytes(zip_path.read_bytes())
            return [target_pdf]

        if not zipfile.is_zipfile(zip_path):
            raise ValueError(f"File is not a valid zip archive: {zip_path}")

        # Target folder for this exam date
        clean_name = re.sub(r"[^\w\-]", "_", exam_date) if exam_date else zip_path.stem
        target_folder = self.extracted_pdfs_dir / clean_name
        target_folder.mkdir(parents=True, exist_ok=True)

        extracted_pdfs = []
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                filename = os.path.basename(member.filename)
                if filename.lower().endswith(".pdf"):
                    # Extract directly into target folder
                    out_path = target_folder / filename
                    with zf.open(member) as src, open(out_path, "wb") as dst:
                        dst.write(src.read())
                    extracted_pdfs.append(out_path)
                    print(f"  -> Extracted: {out_path.name}")

        return extracted_pdfs

    def run_pipeline(
        self,
        limit: Optional[int] = 3,
        force: bool = False,
        target_date: Optional[str] = None,
    ) -> Dict[str, List[Path]]:
        """
        Run the complete scraping pipeline:
        1. Fetch paper list
        2. Filter by target date or limit
        3. Download ZIP archives
        4. Extract shift-wise PDFs
        Returns:
            Dict mapping exam date to list of extracted PDF paths.
        """
        print("[*] Fetching available CPCT question papers from portal...")
        available = self.get_available_papers()
        print(f"[*] Found {len(available)} available exam sessions on portal.")

        if target_date:
            available = [
                p for p in available
                if target_date.lower() in p["exam_date"].lower()
            ]
            if not available:
                print(f"[!] No papers matching date '{target_date}'.")
                return {}

        if limit and limit > 0:
            available = available[:limit]

        results = {}
        for paper in available:
            date_str = paper["exam_date"]
            print(f"\n--- Processing: {date_str} ---")
            try:
                zip_path = self.download_paper(paper, force=force)
                pdfs = self.extract_zip(zip_path, exam_date=date_str)
                results[date_str] = pdfs
            except Exception as e:
                print(f"[!] Error processing {date_str}: {e}")

        total_pdfs = sum(len(p) for p in results.values())
        print(f"\n[DONE] Scraper finished. Extracted {total_pdfs} shift PDFs across {len(results)} exam dates.")
        return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MP CPCT Question Paper Scraper & Unzipper")
    parser.add_argument("--list", action="store_true", help="List available papers on the portal")
    parser.add_argument("--limit", type=int, default=3, help="Number of recent papers to download (default: 3)")
    parser.add_argument("--all", action="store_true", help="Download all available papers")
    parser.add_argument("--date", type=str, help="Filter for specific exam date (e.g. 'Sep 2026')")
    parser.add_argument("--force", action="store_true", help="Force redownload even if file exists")
    args = parser.parse_args()

    scraper = CPCTScraper()
    if args.list:
        papers = scraper.get_available_papers()
        print(f"\nAvailable CPCT Question Papers ({len(papers)} total):")
        print("-" * 75)
        for i, p in enumerate(papers, 1):
            print(f"{i:3d}. Date: {p['exam_date']:<18} | Link: {p['qp_url']}")
        return

    limit = None if args.all else args.limit
    scraper.run_pipeline(limit=limit, force=args.force, target_date=args.date)


if __name__ == "__main__":
    main()
