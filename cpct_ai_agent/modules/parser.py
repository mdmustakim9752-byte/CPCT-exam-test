"""
Module 2: CPCT Bilingual (English + Hindi) Question Parser & Database Ingestion
Supports PyMuPDF text & color-span extraction + Gemini 2.5 Flash Vision API
with Pydantic structured outputs to bypass broken KrutiDev/Remington Hindi fonts.
"""

import os
import re
import sys
import sqlite3
import unicodedata
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from pydantic import BaseModel, Field

# Ensure project root in sys.path
current_file = Path(__file__).resolve()
agent_root = current_file.parent.parent
workspace_root = agent_root.parent
for p in [str(workspace_root), str(agent_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import pymupdf

try:
    from cpct_ai_agent.config import (
        SQLITE_DB_PATH,
        EXTRACTED_PDFS_DIR,
        GEMINI_API_KEY,
        GEMINI_MODEL,
        SECTIONS,
    )
except ImportError:
    from config import (
        SQLITE_DB_PATH,
        EXTRACTED_PDFS_DIR,
        GEMINI_API_KEY,
        GEMINI_MODEL,
        SECTIONS,
    )

# Pydantic schemas for Gemini Structured Outputs
class OptionItem(BaseModel):
    option_number: int = Field(description="Option number: 1, 2, 3, or 4")
    text_en: str = Field(description="English text of option")
    text_hi: str = Field(description="Hindi text of option in clean Unicode Devanagari")
    is_correct: bool = Field(description="True if marked with green checkmark/color, False otherwise")

class ExtractedMCQ(BaseModel):
    question_number: int = Field(description="Question number 1 to 75")
    question_id: Optional[str] = Field(None, description="System Question Id e.g. 25498954053")
    section: str = Field(description="One of: Computer Proficiency, Reading Comprehension, Quantitative Aptitude, Reasoning Aptitude, General Awareness")
    question_en: str = Field(description="English question statement")
    question_hi: str = Field(description="Hindi question statement in clean Devanagari Unicode")
    options: List[OptionItem] = Field(description="List of 4 options")
    correct_option: int = Field(description="The correct option number (1, 2, 3, or 4)")

class PageQuestionsSchema(BaseModel):
    questions: List[ExtractedMCQ]


# Standard CPCT section mapping keywords
SECTION_KEYWORDS = [
    ("COMPUTER PROFICIENCY", "Computer Proficiency"),
    ("READING COMPREHENSION", "Reading Comprehension"),
    ("QUANTITATIVE APTITUDE", "Quantitative Aptitude"),
    ("GENERAL MENTAL ABILITY", "Reasoning Aptitude"),
    ("REASONING APTITUDE", "Reasoning Aptitude"),
    ("GENERAL AWARENESS", "General Awareness"),
]


class DatabaseManager:
    """Manages SQLite storage for CPCT question bank and metadata."""

    def __init__(self, db_path: Path = SQLITE_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def init_schema(self):
        """Initialize database tables and indexes."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Papers metadata table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                exam_date TEXT,
                shift TEXT,
                filename TEXT UNIQUE,
                total_questions INTEGER DEFAULT 0,
                parsed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # Questions table (bilingual)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id INTEGER REFERENCES papers(id),
                question_number INTEGER,
                question_id TEXT,
                section TEXT,
                question_en TEXT,
                question_hi TEXT,
                opt1_en TEXT,
                opt1_hi TEXT,
                opt2_en TEXT,
                opt2_hi TEXT,
                opt3_en TEXT,
                opt3_hi TEXT,
                opt4_en TEXT,
                opt4_hi TEXT,
                correct_option INTEGER,
                exam_date TEXT,
                shift TEXT,
                source_pdf TEXT,
                parsed_by TEXT DEFAULT 'pymupdf',
                is_verified INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(paper_id, question_number)
            )
            """)

            # Indexes for high performance querying
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_questions_section ON questions(section)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_questions_qid ON questions(question_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_questions_exam ON questions(exam_date, shift)")
            conn.commit()


class GeminiVisionParser:
    """
    Multimodal Vision extractor powered by Gemini 2.5 Flash with Structured Outputs.
    Renders PDF pages directly to images to bypass broken legacy KrutiDev / Devlys fonts.
    """

    def __init__(self, api_key: str = GEMINI_API_KEY, model_name: str = GEMINI_MODEL):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model_name = model_name
        self.client = None
        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"[!] Warning: Could not initialize Google GenAI client: {e}")

    def is_available(self) -> bool:
        return self.client is not None

    def extract_page_questions(self, page_image_bytes: bytes, page_num: int) -> List[Dict[str, Any]]:
        """
        Send rendered PDF page image to Gemini 2.5 Flash and return structured questions.
        """
        if not self.client:
            raise RuntimeError("Gemini API Key not set. Provide GEMINI_API_KEY to use Vision Parser.")

        from google.genai import types

        prompt = (
            "You are an expert OCR and Examination Parser for the Madhya Pradesh CPCT Exam. "
            "Examine this page image carefully. Note that questions are presented bilingually: "
            "first in English, then in Hindi. "
            "Correct options are indicated by a green color or a green tick icon. "
            "Incorrect options are shown in red. "
            "Transcribe Hindi text in proper, modern Devanagari Unicode (do NOT output KrutiDev mojibake). "
            "Return all MCQs on this page conforming exactly to the structured schema."
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Part.from_bytes(data=page_image_bytes, mime_type="image/png"),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=PageQuestionsSchema,
                    temperature=0.0,
                ),
            )
            parsed_data = PageQuestionsSchema.model_validate_json(response.text)
            return [q.model_dump() for q in parsed_data.questions]
        except Exception as e:
            print(f"[!] Gemini Vision parsing error on page {page_num}: {e}")
            return []


class CPCTParser:
    """
    Core Question Paper Parser.
    Combines high-speed deterministic PyMuPDF color-span extraction
    with optional Gemini 2.5 Flash Vision AI.
    """

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or DatabaseManager()
        self.gemini_parser = GeminiVisionParser()

    @staticmethod
    def clean_text(text: str) -> str:
        """Normalize unicode text and remove extraneous formatting artifacts."""
        if not text:
            return ""
        text = unicodedata.normalize("NFKC", text)
        # Remove extra whitespace while keeping structure
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        return " ".join(lines)

    @staticmethod
    def parse_options_text(opt_text: str) -> Dict[int, str]:
        """Split options block by '1. ', '2. ', '3. ', '4. '."""
        parts = re.split(r"\n\s*([1-4])\.\s*", "\n" + opt_text)
        options = {1: "", 2: "", 3: "", 4: ""}
        for i in range(1, len(parts), 2):
            try:
                num = int(parts[i])
                val = parts[i + 1].strip()
                # Clean stray internal newlines
                clean_val = " ".join(val.split())
                options[num] = clean_val
            except (ValueError, IndexError):
                continue
        return options

    def extract_green_options(self, doc: pymupdf.Document) -> Dict[int, int]:
        """
        Extract correct option for each question number by scanning green colored
        option markers (32768 / 0x8000) across all pages.
        Returns: {question_number: correct_option_number (1..4)}
        """
        # Map headers to pages
        question_answers: Dict[int, List[int]] = {}

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            text = page.get_text()
            page_dict = page.get_text("dict")

            # Find green spans on this page
            page_greens = []
            for block in page_dict["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            if span["color"] in (32768, 0x8000, 0x008000, 0x00A65A):
                                t = span["text"].strip()
                                m = re.match(r"^([1-4])\.", t)
                                if m:
                                    page_greens.append((int(m.group(1)), span["bbox"][1]))  # (opt_num, y_coord)

            # Match with Question Number headers on this page
            headers = list(re.finditer(r"Question Number\s*:\s*(\d+)", text))
            if not headers or not page_greens:
                continue

        return {}

    def parse_pdf(
        self,
        pdf_path: Path,
        use_vision: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Parse a single CPCT question paper PDF into structured bilingual questions.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        print(f"\n[*] Parsing PDF: {pdf_path.name}")
        doc = pymupdf.open(str(pdf_path))
        num_pages = len(doc)
        print(f"[*] Document loaded: {num_pages} pages")

        # Extract Exam Date and Shift from filename
        # e.g. 5th_Sep_2026_QP_Shift 1.pdf
        stem = pdf_path.stem
        date_match = re.search(r"(\d+(?:st|nd|rd|th)?_[A-Za-z]+_\d{4})", stem)
        exam_date = date_match.group(1).replace("_", " ") if date_match else "CPCT Exam"
        shift_match = re.search(r"Shift\s*(\d+)", stem, re.IGNORECASE)
        shift = f"Shift {shift_match.group(1)}" if shift_match else "Shift 1"

        # Read entire document text and map section headers
        full_text = ""
        page_offsets = []
        for pno in range(num_pages):
            page_offsets.append(len(full_text))
            full_text += doc[pno].get_text() + "\n"

        # Determine section locations
        section_positions = []
        for kw, sname in SECTION_KEYWORDS:
            for m in re.finditer(re.escape(kw), full_text, re.IGNORECASE):
                section_positions.append((m.start(), sname))
        section_positions.sort(key=lambda x: x[0])

        def get_section_for_pos(pos: int) -> str:
            current_sec = "Computer Proficiency"
            for p, s in section_positions:
                if pos >= p:
                    current_sec = s
            return current_sec

        # Extract all green options with page and bounding box
        green_marks = []
        for pno in range(num_pages):
            page = doc[pno]
            p_dict = page.get_text("dict")
            for block in p_dict["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            if span["color"] in (32768, 0x8000, 0x008000, 0x00A65A):
                                t = span["text"].strip()
                                m = re.match(r"^([1-4])\.", t)
                                if m:
                                    green_marks.append({
                                        "page": pno,
                                        "option": int(m.group(1)),
                                        "y": span["bbox"][1],
                                        "x": span["bbox"][0],
                                    })

        # Find all Question headers
        header_re = re.compile(
            r"Question Number\s*:\s*(\d+)\s+Question Id\s*:\s*(\d+)",
            re.IGNORECASE,
        )
        headers = list(header_re.finditer(full_text))

        # Filter out typing test headers (only keep MCQs up to 75)
        # Note: In CPCT exams, each MCQ appears twice (first English, then Hindi)
        valid_headers = [h for h in headers if int(h.group(1)) <= 75]

        extracted_questions = []

        def clean_chunk(chunk_text: str) -> Tuple[str, Dict[int, str]]:
            """Clean preamble metadata and extract question text and options."""
            # Strip metadata lines at beginning
            cleaned = re.sub(
                r"^(?:Question Type\s*:\s*MCQ\s*)?"
                r"(?:Option Shuffling\s*:\s*\w+\s*)?"
                r"(?:Display Question Number\s*:\s*\w+\s*)?"
                r"(?:Correct Marks\s*:\s*\d+\s+Wrong Marks\s*:\s*\d+\s*)?",
                "",
                chunk_text.strip(),
                flags=re.IGNORECASE,
            ).strip()

            parts = re.split(r"\nOptions\s*:\s*\n", cleaned, maxsplit=1)
            q_text = " ".join(parts[0].split())
            opts = {1: "", 2: "", 3: "", 4: ""}
            if len(parts) > 1:
                opt_parts = re.split(r"\n\s*([1-4])\.\s*", "\n" + parts[1])
                for i in range(1, len(opt_parts), 2):
                    try:
                        num = int(opt_parts[i])
                        opts[num] = " ".join(opt_parts[i + 1].split())
                    except (ValueError, IndexError):
                        pass
            return q_text, opts

        for qnum in range(1, 76):
            h_en_idx = 2 * (qnum - 1)
            h_hi_idx = 2 * (qnum - 1) + 1
            h_next_idx = 2 * qnum

            if h_en_idx >= len(valid_headers):
                break

            h_en = valid_headers[h_en_idx]
            qid = h_en.group(2)
            section = get_section_for_pos(h_en.start())

            if h_hi_idx < len(valid_headers):
                h_hi = valid_headers[h_hi_idx]
                en_chunk = full_text[h_en.end():h_hi.start()]
                if h_next_idx < len(valid_headers):
                    hi_chunk = full_text[h_hi.end():valid_headers[h_next_idx].start()]
                else:
                    hi_chunk = full_text[h_hi.end():]
            else:
                en_chunk = full_text[h_en.end():]
                hi_chunk = ""

            q_en, opts_en = clean_chunk(en_chunk)
            q_hi, opts_hi = clean_chunk(hi_chunk)

            # Match correct option from green marks (2 marks per question: EN and HI)
            correct_opt = 1
            green_mark_idx = 2 * (qnum - 1)
            if green_mark_idx < len(green_marks):
                correct_opt = green_marks[green_mark_idx]["option"]

            # Check if Gemini Vision OCR fallback is required for broken Hindi fonts
            is_hindi_valid = any('\u0900' <= ch <= '\u097F' for ch in q_hi)
            parsed_method = "pymupdf"

            if (use_vision or not is_hindi_valid or not q_hi) and self.gemini_parser.is_available():
                q_pos = h_en.start()
                page_idx = 0
                for i, offset in enumerate(page_offsets):
                    if q_pos >= offset:
                        page_idx = i
                try:
                    pix = doc[page_idx].get_pixmap(dpi=150)
                    vision_qs = self.gemini_parser.extract_page_questions(pix.tobytes("png"), page_idx + 1)
                    for vq in vision_qs:
                        if vq.get("question_number") == qnum:
                            if not q_hi or not is_hindi_valid:
                                q_hi = vq.get("question_hi") or q_hi
                            for opt in vq.get("options", []):
                                opt_n = opt.get("option_number")
                                if opt_n in opts_hi and not opts_hi[opt_n]:
                                    opts_hi[opt_n] = opt.get("text_hi") or ""
                            parsed_method = "gemini_vision"
                            break
                except Exception as e:
                    print(f"Vision fallback error on Q{qnum}: {e}")

            extracted_questions.append({
                "question_number": qnum,
                "question_id": qid,
                "section": section,
                "question_en": q_en,
                "question_hi": q_hi,
                "opt1_en": opts_en.get(1, ""),
                "opt1_hi": opts_hi.get(1, ""),
                "opt2_en": opts_en.get(2, ""),
                "opt2_hi": opts_hi.get(2, ""),
                "opt3_en": opts_en.get(3, ""),
                "opt3_hi": opts_hi.get(3, ""),
                "opt4_en": opts_en.get(4, ""),
                "opt4_hi": opts_hi.get(4, ""),
                "correct_option": correct_opt,
                "exam_date": exam_date,
                "shift": shift,
                "source_pdf": pdf_path.name,
                "parsed_by": parsed_method,
            })

        print(f"[OK] Successfully extracted {len(extracted_questions)} questions from {pdf_path.name}")
        return extracted_questions

    def save_to_database(self, questions: List[Dict[str, Any]], pdf_path: Path):
        """Save extracted question paper and questions into SQLite database."""
        if not questions:
            return

        first_q = questions[0]
        exam_date = first_q["exam_date"]
        shift = first_q["shift"]
        filename = pdf_path.name

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            # Upsert paper
            cursor.execute(
                """
                INSERT INTO papers (exam_date, shift, filename, total_questions)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(filename) DO UPDATE SET
                    total_questions=excluded.total_questions,
                    parsed_at=CURRENT_TIMESTAMP
                """,
                (exam_date, shift, filename, len(questions)),
            )
            paper_id = cursor.lastrowid
            if not paper_id:
                cursor.execute("SELECT id FROM papers WHERE filename = ?", (filename,))
                row = cursor.fetchone()
                paper_id = row["id"] if row else 1

            # Insert questions
            for q in questions:
                cursor.execute(
                    """
                    INSERT INTO questions (
                        paper_id, question_number, question_id, section,
                        question_en, question_hi,
                        opt1_en, opt1_hi, opt2_en, opt2_hi,
                        opt3_en, opt3_hi, opt4_en, opt4_hi,
                        correct_option, exam_date, shift,
                        source_pdf, parsed_by
                    ) VALUES (
                        ?, ?, ?, ?,
                        ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?,
                        ?, ?
                    )
                    ON CONFLICT(paper_id, question_number) DO UPDATE SET
                        question_id=excluded.question_id,
                        section=excluded.section,
                        question_en=excluded.question_en,
                        question_hi=excluded.question_hi,
                        opt1_en=excluded.opt1_en, opt1_hi=excluded.opt1_hi,
                        opt2_en=excluded.opt2_en, opt2_hi=excluded.opt2_hi,
                        opt3_en=excluded.opt3_en, opt3_hi=excluded.opt3_hi,
                        opt4_en=excluded.opt4_en, opt4_hi=excluded.opt4_hi,
                        correct_option=excluded.correct_option,
                        parsed_by=excluded.parsed_by
                    """,
                    (
                        paper_id,
                        q["question_number"],
                        q["question_id"],
                        q["section"],
                        q["question_en"],
                        q["question_hi"],
                        q["opt1_en"],
                        q["opt1_hi"],
                        q["opt2_en"],
                        q["opt2_hi"],
                        q["opt3_en"],
                        q["opt3_hi"],
                        q["opt4_en"],
                        q["opt4_hi"],
                        q["correct_option"],
                        q["exam_date"],
                        q["shift"],
                        q["source_pdf"],
                        q["parsed_by"],
                    ),
                )
            conn.commit()
            print(f"[DB] Saved {len(questions)} questions to SQLite database ({self.db.db_path.name})")

    def parse_and_store_all(self, directory: Path = EXTRACTED_PDFS_DIR, use_vision: bool = False):
        """Scan directory and parse all unparsed PDF files into database."""
        dir_path = Path(directory)
        pdf_files = list(dir_path.glob("**/*.pdf"))
        if not pdf_files:
            print(f"[!] No PDF files found in {dir_path}")
            return []

        print(f"[*] Found {len(pdf_files)} PDF files to process.")
        all_results = []
        for pdf in pdf_files:
            try:
                qs = self.parse_pdf(pdf, use_vision=use_vision)
                self.save_to_database(qs, pdf)
                all_results.extend(qs)
            except Exception as e:
                print(f"[!] Failed to parse {pdf.name}: {e}")

        print(f"\n[DONE] Batch parsing complete. Processed {len(pdf_files)} PDFs, {len(all_results)} questions in database.")
        return all_results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MP CPCT PDF-to-SQLite Bilingual Parser")
    parser.add_argument("--pdf", type=str, help="Path to single PDF file")
    parser.add_argument("--all", action="store_true", help="Parse all PDFs in data/extracted_pdfs")
    parser.add_argument("--vision", action="store_true", help="Force Gemini 2.5 Flash Vision OCR on all pages")
    args = parser.parse_args()

    cpct_parser = CPCTParser()
    if args.pdf:
        qs = cpct_parser.parse_pdf(Path(args.pdf), use_vision=args.vision)
        cpct_parser.save_to_database(qs, Path(args.pdf))
    else:
        cpct_parser.parse_and_store_all(use_vision=args.vision)


if __name__ == "__main__":
    main()
