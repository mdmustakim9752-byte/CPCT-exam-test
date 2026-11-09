"""
CPCT AI Exam Scraper, Trend Analyzer & Personal Trainer Agent
Configuration Module
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Load .env file from either BASE_DIR or PROJECT_ROOT
load_dotenv(BASE_DIR / ".env")
load_dotenv(PROJECT_ROOT / ".env")

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_ZIPS_DIR = DATA_DIR / "raw_zips"
EXTRACTED_PDFS_DIR = DATA_DIR / "extracted_pdfs"
DB_DIR = DATA_DIR / "db"

# Database storage paths
SQLITE_DB_PATH = DB_DIR / "cpct_questions.db"
CHROMA_DIR = DB_DIR / "chromadb"
CHROMA_COLLECTION_NAME = "cpct_questions"

# Portal URLs & scraping settings
CPCT_PORTAL_URL = "https://www.cpct.mp.gov.in/per/g01/pub/1172/ASM/WebPortal/1/Links.html"
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,hi;q=0.8",
}
REQUEST_TIMEOUT = 30  # seconds

# Google Gemini API Settings - Reload from .env dynamically
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Analysis & Vector Similarity settings
SIMILARITY_REPEAT_THRESHOLD = float(os.getenv("SIMILARITY_REPEAT_THRESHOLD", "0.82"))

# Standard CPCT Exam Syllabus Sections
SECTIONS = [
    "Computer Proficiency",
    "Reading Comprehension",
    "Quantitative Aptitude",
    "Reasoning Aptitude",
    "General Awareness",
]

# Section weights in actual MP CPCT exam (Total 75 MCQs)
SECTION_WEIGHTS = {
    "Computer Proficiency": 52,
    "Reading Comprehension": 5,
    "Quantitative Aptitude": 6,
    "Reasoning Aptitude": 6,
    "General Awareness": 6,
}

def ensure_directories():
    """Ensure all required project directories exist."""
    for path in [DATA_DIR, RAW_ZIPS_DIR, EXTRACTED_PDFS_DIR, DB_DIR, CHROMA_DIR]:
        path.mkdir(parents=True, exist_ok=True)

# Run on import to ensure directories exist
ensure_directories()

