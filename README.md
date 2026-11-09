# MP CPCT AI Exam Scraper, Trend Analyzer & Personal Trainer Agent 🎯

A complete, modular AI Agent system that automatically scrapes official Madhya Pradesh CPCT (Computer Proficiency Certification Test) previous year question papers, extracts bilingual (English + Hindi) questions, identifies high-probability repeating questions using ChromaDB vector similarity, and provides an interactive AI Personal Trainer coach with conversational Hinglish explanations.

---

## 🌟 Key Features

1. **Auto-Downloader & Unzipper (`modules/scraper.py`)**:
   - Scrapes the live MP CPCT portal (`cpct.mp.gov.in`).
   - Discovers all 130+ historical exam cycles (2015–2026).
   - Automatically downloads ZIP files and extracts shift-wise PDFs.

2. **Bilingual PDF & Vision Parser (`modules/parser.py`)**:
   - High-precision PyMuPDF text & color-span extraction (`0x8000` green marker ground-truth answer key).
   - Google Gemini 2.5 Flash Vision integration with strict Pydantic JSON schemas to bypass broken legacy KrutiDev/Remington Hindi fonts.
   - Saves structured questions directly into SQLite (`data/db/cpct_questions.db`).

3. **ChromaDB Trend Analyzer & Repeat Detector (`modules/analyzer.py`)**:
   - Converts CPCT questions into dense vector embeddings.
   - Detects semantically rephrased repeating questions across shifts using cosine similarity.
   - Calculates **Repeat Probability Scores** (up to 98%) and section weightage.
   - Provides instant **Semantic Vector Search** across the entire question bank.

4. **Adaptive Personal Trainer Coach (`modules/trainer.py`)**:
   - **Weakness Radar**: Automatically serves questions from topics where user accuracy is lowest.
   - **High-Yield Repeat Drill**: Prioritizes questions with the highest recurrence rates.
   - **Hinglish AI Explanation Engine**: Explains core concepts, why options are wrong, and exam hacks in friendly Hinglish (powered by Gemini 2.5 Flash with offline fallback).
   - **Full 75-Question Mock Exam**: Simulates official CPCT exam proportions (52 Computer, 5 Reading Comprehension, 6 Quantitative, 6 Reasoning, 6 General Awareness).

5. **Dual Interface (`main.py`)**:
   - **Interactive Terminal CLI** with rich tables and live quiz.
   - **Modern Web Application** (FastAPI + Tailwind CSS + Alpine.js) served at `http://localhost:8000`.

---

## 📁 Project Directory Structure

```text
cpct_ai_agent/
├── data/
│   ├── raw_zips/          # Downloaded ZIP files from CPCT portal
│   ├── extracted_pdfs/    # Unzipped shift-wise PDF papers
│   └── db/                # SQLite (cpct_questions.db) and ChromaDB persistent storage
├── modules/
│   ├── scraper.py         # Module 1: Auto-downloader & Unzipper
│   ├── parser.py          # Module 2: Gemini Vision / PyMuPDF bilingual extractor
│   ├── analyzer.py        # Module 3: ChromaDB clustering & high-yield detector
│   └── trainer.py         # Module 4: Adaptive quiz & Hinglish explanation engine
├── main.py                # CLI & FastAPI web server entry point
├── config.py              # Central configuration & paths
├── requirements.txt       # Python dependencies
└── .env                   # Gemini API key and settings
```

---

## 🚀 Quick Start Guide

### 1. Activate Environment & Configure API Key
```bash
# Activate existing virtual environment
.venv\Scripts\activate

# Add your Gemini API Key in .env (Optional - fallback is active offline)
cp .env.example .env
```

### 2. Launch the Web Application
```bash
python main.py serve --port 8000
```
Open your browser at **`http://localhost:8000`** to access:
- 📊 **Dashboard**: Total questions, ChromaDB vectors, and section analytics.
- 🔥 **Repeat Trends**: Top repeating questions with probability bars.
- 🎯 **AI Trainer**: Interactive bilingual quiz with instant Hinglish explanations.
- 🔍 **Semantic Search**: Search questions by concept (e.g., *"compiler vs interpreter"*).
- 📥 **Portal Scraper**: 1-click download and parsing of new papers.

---

### 3. CLI Subcommands

```bash
# Check system status
python main.py status

# Download latest 2 CPCT papers from portal
python main.py scrape --limit 2

# Parse all downloaded PDFs into SQLite & ChromaDB
python main.py parse --all

# Analyze high-yield repeating questions
python main.py analyze --top 10

# Search questions semantically
python main.py analyze --search "keyboard shortcut for save"

# Launch terminal interactive quiz
python main.py train
```

