"""
CPCT AI Exam Scraper, Trend Analyzer & Personal Trainer Agent
Main CLI & FastAPI Entry Point
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure project root in sys.path
current_file = Path(__file__).resolve()
agent_root = current_file.parent
workspace_root = agent_root.parent
for p in [str(workspace_root), str(agent_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from cpct_ai_agent.config import (
    SQLITE_DB_PATH,
    CHROMA_DIR,
    RAW_ZIPS_DIR,
    EXTRACTED_PDFS_DIR,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    SECTIONS,
)
from cpct_ai_agent.modules.scraper import CPCTScraper
from cpct_ai_agent.modules.parser import CPCTParser
from cpct_ai_agent.modules.analyzer import CPCTAnalyzer
from cpct_ai_agent.modules.trainer import CPCTTrainer

# Initialize FastAPI App
app = FastAPI(
    title="MP CPCT AI Agent API",
    description="Automated Scraper, Bilingual Extractor, Vector Trend Analyzer & Personal Trainer for MP CPCT Exams",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared singletons
scraper = CPCTScraper()
parser = CPCTParser()
analyzer = CPCTAnalyzer()
trainer = CPCTTrainer()


# ==========================================
# FASTAPI REST ENDPOINTS
# ==========================================

@app.get("/api/status")
def get_system_status():
    """Get system health, parsed question counts, and vector store stats."""
    import sqlite3
    db_exists = SQLITE_DB_PATH.exists()
    total_questions = 0
    total_papers = 0
    section_breakdown = {}

    if db_exists:
        try:
            with sqlite3.connect(str(SQLITE_DB_PATH)) as conn:
                r_q = conn.execute("SELECT count(*) FROM questions").fetchone()
                total_questions = r_q[0] if r_q else 0
                r_p = conn.execute("SELECT count(*) FROM papers").fetchone()
                total_papers = r_p[0] if r_p else 0
                r_s = conn.execute("SELECT section, count(*) FROM questions GROUP BY section").fetchall()
                section_breakdown = {s[0]: s[1] for s in r_s}
        except Exception:
            pass

    chroma_count = analyzer.collection.count()
    zips_count = len(list(RAW_ZIPS_DIR.glob("*.zip")))
    pdfs_count = len(list(EXTRACTED_PDFS_DIR.glob("**/*.pdf")))
    user_perf = trainer.get_user_performance_summary()

    return {
        "status": "healthy",
        "gemini_configured": bool(GEMINI_API_KEY),
        "gemini_model": GEMINI_MODEL,
        "total_papers_parsed": total_papers,
        "total_questions_in_db": total_questions,
        "chroma_embeddings_count": chroma_count,
        "raw_zips_count": zips_count,
        "extracted_pdfs_count": pdfs_count,
        "section_breakdown": section_breakdown,
        "user_performance": user_perf,
    }


@app.get("/api/papers/available")
def get_available_papers():
    """Fetch live list of exam cycles from the CPCT portal."""
    try:
        papers = scraper.get_available_papers()
        return {"total": len(papers), "papers": papers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ScrapeRequest(BaseModel):
    limit: Optional[int] = 2
    exam_date: Optional[str] = None
    force: bool = False


@app.post("/api/scraper/download")
def trigger_download_and_extract(req: ScrapeRequest):
    """Download and extract question paper ZIPs from portal."""
    try:
        results = scraper.run_pipeline(limit=req.limit, target_date=req.exam_date, force=req.force)
        return {
            "message": "Scraping completed successfully",
            "sessions": {k: [str(p.name) for p in v] for k, v in results.items()},
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ParseRequest(BaseModel):
    use_vision: bool = False


@app.post("/api/parser/run")
def trigger_batch_parse(req: ParseRequest):
    """Parse all unparsed PDF files into SQLite database."""
    try:
        parsed = parser.parse_and_store_all(use_vision=req.use_vision)
        # Sync newly parsed questions into ChromaDB
        analyzer.sync_to_chroma()
        analyzer.detect_high_yield_questions()
        return {
            "message": "Parsing and Vector indexing complete",
            "total_extracted": len(parsed),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/trends/high-yield")
def get_high_yield_trends(
    limit: int = Query(20, ge=1, le=100),
    section: Optional[str] = None,
):
    """Get high-probability repeating questions."""
    trends = analyzer.get_high_yield_report(limit=limit, section=section)
    return {"count": len(trends), "questions": trends}


@app.get("/api/trends/distribution")
def get_topic_distribution():
    """Get section-wise repeat frequency distribution."""
    return {"distribution": analyzer.get_topic_distribution()}


@app.get("/api/search")
def semantic_search(
    query: str,
    limit: int = Query(5, ge=1, le=20),
    section: Optional[str] = None,
):
    """Semantic vector search across CPCT question bank."""
    results = analyzer.semantic_search(query, n_results=limit, section=section)
    return {"query": query, "count": len(results), "results": results}


class ChatAskRequest(BaseModel):
    question_id: int
    doubt_query: str

@app.post("/api/chat/ask")
def ask_ai_coach(req: ChatAskRequest):
    """Ask follow-up doubts directly to AI Coach powered by Gemini 2.5 Flash."""
    try:
        import sqlite3
        with sqlite3.connect(str(SQLITE_DB_PATH)) as conn:
            conn.row_factory = sqlite3.Row
            q = conn.execute("SELECT * FROM questions WHERE id = ?", (req.question_id,)).fetchone()
            if not q:
                raise HTTPException(status_code=404, detail="Question not found")
            q_dict = dict(q)

        answer = trainer.explainer.ask_doubt(q_dict, req.doubt_query)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/trainer/reset")
def reset_trainer_scorecard():
    """Reset user attempts, accuracy, and mastery scorecard."""
    try:
        trainer.reset_user_stats()
        return {"message": "Scorecard and attempts successfully reset."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/export/questions")
def export_questions_json():
    """Export the entire question bank as a JSON file."""
    import sqlite3
    with sqlite3.connect(str(SQLITE_DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM questions ORDER BY id ASC").fetchall()
        return [dict(r) for r in rows]


@app.get("/api/quiz/next")
def get_quiz_question(
    mode: str = Query("adaptive", pattern="^(adaptive|high_yield|section|random)$"),
    section: Optional[str] = None,
):
    """Fetch next question for personal trainer quiz."""
    q = trainer.get_next_question(mode=mode, section=section)
    if not q:
        raise HTTPException(status_code=404, detail="No questions available matching criteria.")
    return q


class QuizSubmitRequest(BaseModel):
    question_id: int
    selected_option: int
    time_spent_seconds: float = 0.0


@app.post("/api/quiz/submit")
def submit_quiz_answer(req: QuizSubmitRequest):
    """Submit user answer, get evaluation, updated mastery, and Hinglish explanation."""
    try:
        result = trainer.record_attempt(
            question_id=req.question_id,
            selected_option=req.selected_option,
            time_spent_seconds=req.time_spent_seconds,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/quiz/stats")
def get_user_stats():
    """Fetch user mastery scorecard and accuracy stats."""
    return trainer.get_user_performance_summary()


@app.get("/api/mock-exam")
def generate_mock_exam():
    """Generate official 75-question Mock Exam."""
    questions = trainer.generate_mock_exam()
    return {"total": len(questions), "questions": questions}


# ==========================================
# MODERN WEB UI (HTML / Tailwind / Alpine.js)
# ==========================================

WEB_UI_HTML = """
<!DOCTYPE html>
<html lang="hi" class="h-full">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MP CPCT AI Exam Master 🎓 | Scraper, Trend Analyzer & Personal Trainer</title>
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Alpine.js -->
    <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
    <!-- FontAwesome 6 -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Noto+Sans+Devanagari:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        [x-cloak] { display: none !important; }
        body {
            font-family: 'Plus Jakarta Sans', 'Noto Sans Devanagari', sans-serif;
        }
        .gradient-brand {
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 35%, #1e3a8a 70%, #0284c7 100%);
        }
        .glass-panel {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(226, 232, 240, 0.8);
        }
        .glass-card-dark {
            background: rgba(15, 23, 42, 0.85);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .pulsing-dot {
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
            animation: pulse 1.8s infinite cubic-bezier(0.66, 0, 0, 1);
        }
        @keyframes pulse {
            to { box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
        }
        /* Custom scrollbars */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: #f1f5f9; }
        ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 9999px; }
        ::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
    </style>
</head>
<body class="bg-slate-100/80 text-slate-900 min-h-screen flex flex-col antialiased selection:bg-blue-600 selection:text-white"
      x-data="cpctApp()" x-init="initApp()">

    <!-- ==================== HEADER / NAVBAR ==================== -->
    <header class="gradient-brand text-white sticky top-0 z-50 shadow-xl border-b border-white/10">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 py-3.5 flex flex-col lg:flex-row items-center justify-between gap-4">
            
            <!-- Logo & Brand Badge -->
            <div class="flex items-center space-x-3 w-full lg:w-auto justify-between lg:justify-start">
                <div class="flex items-center space-x-3">
                    <div class="w-11 h-11 rounded-2xl bg-gradient-to-tr from-amber-400 to-yellow-300 flex items-center justify-center text-slate-950 font-black shadow-lg shadow-amber-500/20 text-xl">
                        <i class="fa-solid fa-graduation-cap"></i>
                    </div>
                    <div>
                        <div class="flex items-center gap-2">
                            <h1 class="text-xl font-extrabold tracking-tight text-white">MP CPCT AI Master</h1>
                            <span class="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                                <span class="w-2 h-2 rounded-full bg-emerald-400 pulsing-dot"></span>
                                Gemini 2.5 Flash
                            </span>
                        </div>
                        <p class="text-xs text-blue-200/90 font-medium">Bilingual Scraper • ChromaDB Trend Analyzer • Personal Trainer Coach</p>
                    </div>
                </div>

                <!-- Sound & Quick Controls (Mobile visible) -->
                <div class="flex items-center gap-2 lg:hidden">
                    <button @click="toggleSound()" class="p-2 rounded-xl bg-white/10 text-white hover:bg-white/20 transition text-sm">
                        <i :class="soundEnabled ? 'fa-solid fa-volume-high text-emerald-400' : 'fa-solid fa-volume-xmark text-slate-400'"></i>
                    </button>
                </div>
            </div>

            <!-- Main Navigation Tabs -->
            <nav class="flex flex-wrap items-center justify-center gap-1.5 bg-black/30 p-1.5 rounded-2xl border border-white/10 text-xs sm:text-sm font-semibold w-full lg:w-auto overflow-x-auto">
                <button @click="switchTab('dashboard')" :class="activeTab === 'dashboard' ? 'bg-white text-slate-900 shadow-md' : 'text-slate-200 hover:bg-white/10'" class="px-3.5 py-2 rounded-xl transition flex items-center gap-2">
                    <i class="fa-solid fa-chart-pie text-blue-600"></i> डैशबोर्ड
                </button>
                <button @click="switchTab('high_yield')" :class="activeTab === 'high_yield' ? 'bg-white text-slate-900 shadow-md' : 'text-slate-200 hover:bg-white/10'" class="px-3.5 py-2 rounded-xl transition flex items-center gap-2">
                    <i class="fa-solid fa-fire text-amber-500"></i> रिपीट प्रश्न
                </button>
                <button @click="switchTab('quiz')" :class="activeTab === 'quiz' ? 'bg-white text-slate-900 shadow-md' : 'text-slate-200 hover:bg-white/10'" class="px-3.5 py-2 rounded-xl transition flex items-center gap-2">
                    <i class="fa-solid fa-bullseye text-emerald-500"></i> AI ट्रेनर कोच
                </button>
                <button @click="switchTab('mock_exam')" :class="activeTab === 'mock_exam' ? 'bg-white text-slate-900 shadow-md' : 'text-slate-200 hover:bg-white/10'" class="px-3.5 py-2 rounded-xl transition flex items-center gap-2">
                    <i class="fa-solid fa-stopwatch text-indigo-500"></i> असली मॉक टेस्ट (75 Qs)
                </button>
                <button @click="switchTab('search')" :class="activeTab === 'search' ? 'bg-white text-slate-900 shadow-md' : 'text-slate-200 hover:bg-white/10'" class="px-3.5 py-2 rounded-xl transition flex items-center gap-2">
                    <i class="fa-solid fa-magnifying-glass text-cyan-500"></i> सिमेंटिक सर्च
                </button>
                <button @click="switchTab('control')" :class="activeTab === 'control' ? 'bg-white text-slate-900 shadow-md' : 'text-slate-200 hover:bg-white/10'" class="px-3.5 py-2 rounded-xl transition flex items-center gap-2">
                    <i class="fa-solid fa-gears text-purple-400"></i> कंट्रोल सेंटर
                </button>
            </nav>

            <!-- Sound toggle (Desktop) -->
            <div class="hidden lg:flex items-center gap-2.5">
                <button @click="toggleSound()" class="px-3 py-1.5 rounded-xl bg-white/10 hover:bg-white/20 text-white text-xs font-semibold flex items-center gap-2 transition" :title="soundEnabled ? 'ध्वनि प्रभाव चालू हैं' : 'ध्वनि प्रभाव बंद हैं'">
                    <i :class="soundEnabled ? 'fa-solid fa-volume-high text-emerald-400' : 'fa-solid fa-volume-xmark text-slate-400'"></i>
                    <span x-text="soundEnabled ? 'Audio ON' : 'Audio OFF'"></span>
                </button>
            </div>
        </div>
    </header>

    <!-- ==================== MAIN CONTENT CONTAINER ==================== -->
    <main class="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-6 md:py-8">

        <!-- ======================================================== -->
        <!-- TAB 1: DASHBOARD (डैशबोर्ड & संपूर्ण स्थिति) -->
        <!-- ======================================================== -->
        <section x-show="activeTab === 'dashboard'" x-cloak class="space-y-6">
            <!-- Hero Banner -->
            <div class="glass-panel p-6 sm:p-8 rounded-3xl shadow-sm border border-slate-200/80 bg-gradient-to-r from-blue-900 via-indigo-900 to-slate-900 text-white relative overflow-hidden">
                <div class="relative z-10 max-w-2xl">
                    <span class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-amber-400/20 text-amber-300 border border-amber-400/30 mb-3">
                        <i class="fa-solid fa-sparkles"></i> AI Powered Exam Coach
                    </span>
                    <h2 class="text-2xl sm:text-3xl font-extrabold tracking-tight mb-2">मध्य प्रदेश CPCT परीक्षा में 100% सफलता</h2>
                    <p class="text-blue-100 text-sm sm:text-base leading-relaxed mb-5">
                        पिछले वर्षों के सभी पेपर्स का ऑटोमेटेड विश्लेषण, बार-बार पूछे जाने वाले <strong class="text-amber-300">High-Probability प्रश्न</strong>, और हिंग्लिश में सरल स्पष्टीकरण।
                    </p>
                    <div class="flex flex-wrap gap-3">
                        <button @click="switchTab('quiz')" class="px-5 py-2.5 bg-gradient-to-r from-amber-400 to-yellow-400 text-slate-950 rounded-xl font-bold text-sm shadow-lg shadow-amber-500/20 hover:scale-105 active:scale-95 transition flex items-center gap-2">
                            <i class="fa-solid fa-play"></i> आज का अभ्यास शुरू करें
                        </button>
                        <button @click="switchTab('high_yield')" class="px-5 py-2.5 bg-white/10 hover:bg-white/20 text-white rounded-xl font-bold text-sm border border-white/20 transition flex items-center gap-2">
                            <i class="fa-solid fa-fire text-amber-400"></i> टॉप 25 रिपीट प्रश्न देखें
                        </button>
                    </div>
                </div>
                <!-- Background decoration icon -->
                <i class="fa-solid fa-computer absolute -right-8 -bottom-10 text-9xl text-white/5 pointer-events-none"></i>
            </div>

            <!-- Top Statistics Counter Grid -->
            <div class="grid grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-5">
                <div class="glass-panel p-5 rounded-2xl shadow-sm border border-slate-200 flex items-center justify-between">
                    <div>
                        <p class="text-xs font-bold text-slate-500 uppercase tracking-wider">कुल प्रश्न (Questions)</p>
                        <h3 class="text-2xl sm:text-3xl font-black text-blue-600 mt-1" x-text="status.total_questions_in_db || 0">0</h3>
                        <p class="text-xs text-slate-500 mt-0.5">Bilingual + Answer Key</p>
                    </div>
                    <div class="w-12 h-12 bg-blue-100 text-blue-600 rounded-2xl flex items-center justify-center text-xl shrink-0">
                        <i class="fa-solid fa-database"></i>
                    </div>
                </div>

                <div class="glass-panel p-5 rounded-2xl shadow-sm border border-slate-200 flex items-center justify-between">
                    <div>
                        <p class="text-xs font-bold text-slate-500 uppercase tracking-wider">ChromaDB वेक्टर्स</p>
                        <h3 class="text-2xl sm:text-3xl font-black text-cyan-600 mt-1" x-text="status.chroma_embeddings_count || 0">0</h3>
                        <p class="text-xs text-slate-500 mt-0.5">सिमेंटिक एम्बेडिंग्स</p>
                    </div>
                    <div class="w-12 h-12 bg-cyan-100 text-cyan-600 rounded-2xl flex items-center justify-center text-xl shrink-0">
                        <i class="fa-solid fa-network-wired"></i>
                    </div>
                </div>

                <div class="glass-panel p-5 rounded-2xl shadow-sm border border-slate-200 flex items-center justify-between">
                    <div>
                        <p class="text-xs font-bold text-slate-500 uppercase tracking-wider">हल किए गए शिफ्ट पेपर्स</p>
                        <h3 class="text-2xl sm:text-3xl font-black text-indigo-600 mt-1" x-text="status.total_papers_parsed || 0">0</h3>
                        <p class="text-xs text-slate-500 mt-0.5">MP CPCT Official Shifts</p>
                    </div>
                    <div class="w-12 h-12 bg-indigo-100 text-indigo-600 rounded-2xl flex items-center justify-center text-xl shrink-0">
                        <i class="fa-solid fa-file-pdf"></i>
                    </div>
                </div>

                <div class="glass-panel p-5 rounded-2xl shadow-sm border border-slate-200 flex items-center justify-between">
                    <div>
                        <p class="text-xs font-bold text-slate-500 uppercase tracking-wider">आपकी सटीकता (Accuracy)</p>
                        <h3 class="text-2xl sm:text-3xl font-black text-emerald-600 mt-1" x-text="(status.user_performance?.overall_accuracy || 0) + '%'">0%</h3>
                        <p class="text-xs text-slate-500 mt-0.5"><span x-text="status.user_performance?.total_attempted || 0">0</span> प्रश्न हल किए</p>
                    </div>
                    <div class="w-12 h-12 bg-emerald-100 text-emerald-600 rounded-2xl flex items-center justify-center text-xl shrink-0">
                        <i class="fa-solid fa-award"></i>
                    </div>
                </div>
            </div>

            <!-- Two Column: Section Weightage & Weakness Radar -->
            <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <!-- Section Weightage Breakdown -->
                <div class="glass-panel p-6 rounded-3xl shadow-sm border border-slate-200">
                    <div class="flex items-center justify-between mb-4">
                        <h3 class="text-base font-bold text-slate-800 flex items-center gap-2">
                            <i class="fa-solid fa-chart-column text-blue-600"></i> CPCT परीक्षा सिलेबस वेटेज
                        </h3>
                        <span class="text-xs font-semibold text-slate-500">कुल 75 प्रश्न</span>
                    </div>

                    <div class="space-y-4">
                        <div class="p-3 bg-blue-50/70 border border-blue-200 rounded-xl">
                            <div class="flex justify-between text-xs font-bold text-blue-900 mb-1">
                                <span>1. कम्प्यूटर दक्षता (Computer IT)</span>
                                <span>52 प्रश्न (69.3%)</span>
                            </div>
                            <div class="w-full bg-blue-200 rounded-full h-2">
                                <div class="bg-blue-600 h-2 rounded-full" style="width: 69.3%"></div>
                            </div>
                            <p class="text-[11px] text-blue-700 mt-1 font-medium">💡 सबसे महत्वपूर्ण सेक्शन! इसे पास करना अनिवार्य है।</p>
                        </div>

                        <div class="p-3 bg-slate-50 border border-slate-200 rounded-xl">
                            <div class="flex justify-between text-xs font-bold text-slate-800 mb-1">
                                <span>2. क्वांटिटेटिव एप्टीट्यूड (गणित)</span>
                                <span>6 प्रश्न (8%)</span>
                            </div>
                            <div class="w-full bg-slate-200 rounded-full h-2">
                                <div class="bg-indigo-600 h-2 rounded-full" style="width: 8%"></div>
                            </div>
                        </div>

                        <div class="p-3 bg-slate-50 border border-slate-200 rounded-xl">
                            <div class="flex justify-between text-xs font-bold text-slate-800 mb-1">
                                <span>3. रीजनिंग एबिलिटी (तर्कशक्ति)</span>
                                <span>6 प्रश्न (8%)</span>
                            </div>
                            <div class="w-full bg-slate-200 rounded-full h-2">
                                <div class="bg-indigo-600 h-2 rounded-full" style="width: 8%"></div>
                            </div>
                        </div>

                        <div class="p-3 bg-slate-50 border border-slate-200 rounded-xl">
                            <div class="flex justify-between text-xs font-bold text-slate-800 mb-1">
                                <span>4. सामान्य जागरूकता (GK / GA)</span>
                                <span>6 प्रश्न (8%)</span>
                            </div>
                            <div class="w-full bg-slate-200 rounded-full h-2">
                                <div class="bg-indigo-600 h-2 rounded-full" style="width: 8%"></div>
                            </div>
                        </div>

                        <div class="p-3 bg-slate-50 border border-slate-200 rounded-xl">
                            <div class="flex justify-between text-xs font-bold text-slate-800 mb-1">
                                <span>5. रीडिंग कॉम्प्रिहेंशन (गद्यांश)</span>
                                <span>5 प्रश्न (6.7%)</span>
                            </div>
                            <div class="w-full bg-slate-200 rounded-full h-2">
                                <div class="bg-indigo-600 h-2 rounded-full" style="width: 6.7%"></div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Weakness & Readiness Radar -->
                <div class="glass-panel p-6 rounded-3xl shadow-sm border border-slate-200 lg:col-span-2 flex flex-col justify-between">
                    <div>
                        <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4">
                            <div>
                                <h3 class="text-base font-bold text-slate-800 flex items-center gap-2">
                                    <i class="fa-solid fa-crosshairs text-emerald-600"></i> आपकी विषय-वार तैयारी (Section Mastery)
                                </h3>
                                <p class="text-xs text-slate-500">कमजोर विषयों पर हमारा AI ट्रेनर अपने-आप ज्यादा ध्यान देगा।</p>
                            </div>
                            <span class="px-3 py-1 bg-emerald-100 text-emerald-800 text-xs font-bold rounded-full w-fit">
                                पासिंग कटऑफ: 38 अंक (50%)
                            </span>
                        </div>

                        <div class="overflow-x-auto">
                            <table class="w-full text-left text-sm">
                                <thead class="bg-slate-50 text-slate-500 font-bold text-xs uppercase border-b border-slate-200">
                                    <tr>
                                        <th class="py-2.5 px-3">सेक्शन का नाम</th>
                                        <th class="py-2.5 px-3 text-center">हल किए गए</th>
                                        <th class="py-2.5 px-3 text-center">सटीकता (%)</th>
                                        <th class="py-2.5 px-3 text-right">तैयारी का स्तर</th>
                                    </tr>
                                </thead>
                                <tbody class="divide-y divide-slate-100">
                                    <template x-for="item in status.user_performance?.sections || []" :key="item.section">
                                        <tr class="hover:bg-slate-50/70 transition">
                                            <td class="py-3 px-3 font-semibold text-slate-800 flex items-center gap-2">
                                                <i class="fa-solid fa-circle text-[8px]" :class="item.accuracy >= 70 ? 'text-emerald-500' : (item.accuracy >= 50 ? 'text-amber-500' : 'text-rose-500')"></i>
                                                <span x-text="item.section"></span>
                                            </td>
                                            <td class="py-3 px-3 text-center font-medium text-slate-600" x-text="item.total_attempts"></td>
                                            <td class="py-3 px-3 text-center">
                                                <span class="font-extrabold" :class="item.accuracy >= 70 ? 'text-emerald-600' : (item.accuracy >= 50 ? 'text-amber-600' : 'text-rose-600')" x-text="item.total_attempts > 0 ? item.accuracy + '%' : '-'"></span>
                                            </td>
                                            <td class="py-3 px-3 text-right">
                                                <span class="px-2.5 py-1 text-xs font-bold rounded-full inline-block"
                                                    :class="{
                                                        'bg-emerald-100 text-emerald-800': item.mastery_level === 'Master' || item.mastery_level === 'Exam Ready',
                                                        'bg-amber-100 text-amber-800': item.mastery_level === 'Intermediate',
                                                        'bg-rose-100 text-rose-800': item.mastery_level === 'Needs Practice' || item.mastery_level === 'Beginner'
                                                    }"
                                                    x-text="item.mastery_level === 'Exam Ready' ? 'तैयार ✅' : (item.mastery_level === 'Master' ? 'मास्टर 🏆' : (item.mastery_level === 'Intermediate' ? 'मध्यम ⚠️' : 'अभ्यास जरूरी ❌'))">
                                                </span>
                                            </td>
                                        </tr>
                                    </template>
                                </tbody>
                            </table>
                        </div>
                    </div>

                    <!-- Quick Practice Prompt -->
                    <div class="mt-6 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-2xl border border-blue-100 flex flex-col sm:flex-row items-center justify-between gap-3">
                        <div class="flex items-center gap-3">
                            <div class="w-10 h-10 rounded-xl bg-blue-600 text-white flex items-center justify-center font-bold shrink-0">
                                <i class="fa-solid fa-wand-magic-sparkles"></i>
                            </div>
                            <div>
                                <p class="text-xs font-bold text-slate-800">कमजोर सेक्शन सुधारें: <span class="text-blue-700" x-text="status.user_performance?.weakest_section || 'Computer Proficiency'"></span></p>
                                <p class="text-[11px] text-slate-500">AI कोच ने आपके लिए व्यक्तिगत अभ्यास तैयार किया है।</p>
                            </div>
                        </div>
                        <button @click="startAdaptiveSectionDrill(status.user_performance?.weakest_section)" class="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white font-bold text-xs rounded-xl transition shrink-0">
                            तुरंत हल करें &rarr;
                        </button>
                    </div>
                </div>
            </div>
        </section>

        <!-- ======================================================== -->
        <!-- TAB 2: HIGH-YIELD REPEAT QUESTIONS (रिपीट प्रश्न ट्रेंड) -->
        <!-- ======================================================== -->
        <section x-show="activeTab === 'high_yield'" x-cloak class="space-y-6">
            <div class="glass-panel p-6 rounded-3xl shadow-sm border border-slate-200 flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                    <h2 class="text-xl sm:text-2xl font-black text-slate-800 flex items-center gap-2">
                        <i class="fa-solid fa-fire text-amber-500"></i> बार-बार रिपीट होने वाले प्रश्न (High-Yield Questions)
                    </h2>
                    <p class="text-slate-500 text-xs sm:text-sm mt-1">
                        ChromaDB वेक्टर समानता द्वारा पिछले विभिन्न वर्षों के पेपर्स में पूछे गए सबसे महत्वपूर्ण प्रश्न।
                    </p>
                </div>
                <!-- Filter by section -->
                <div class="flex items-center gap-2">
                    <label class="text-xs font-bold text-slate-600">सेक्शन चुनें:</label>
                    <select x-model="filterSection" @change="loadHighYield()" class="bg-white border border-slate-300 rounded-xl px-3 py-2 text-xs font-bold text-slate-700 shadow-sm outline-none focus:ring-2 focus:ring-blue-500">
                        <option value="">सभी विषय (All Sections)</option>
                        <option value="Computer Proficiency">💻 Computer Proficiency (52 Qs)</option>
                        <option value="Reading Comprehension">📖 Reading Comprehension (5 Qs)</option>
                        <option value="Quantitative Aptitude">🔢 Quantitative Aptitude (6 Qs)</option>
                        <option value="Reasoning Aptitude">🧩 Reasoning Aptitude (6 Qs)</option>
                        <option value="General Awareness">🌍 General Awareness (6 Qs)</option>
                    </select>
                </div>
            </div>

            <!-- Question Cards List -->
            <div class="space-y-4">
                <template x-for="(q, idx) in highYieldQuestions" :key="q.id">
                    <div class="glass-panel p-5 sm:p-6 rounded-3xl shadow-sm border border-slate-200/90 hover:shadow-md transition">
                        <!-- Top Meta Header -->
                        <div class="flex flex-wrap items-center justify-between gap-2 mb-3">
                            <div class="flex items-center gap-2">
                                <span class="bg-blue-600 text-white text-xs font-extrabold px-3 py-1 rounded-lg" x-text="'Q#' + q.question_number"></span>
                                <span class="bg-slate-100 text-slate-700 text-xs font-bold px-3 py-1 rounded-lg border border-slate-200" x-text="q.section"></span>
                                <span class="text-xs text-slate-400 font-medium" x-text="q.exam_date + ' • ' + q.shift"></span>
                            </div>

                            <div class="flex items-center gap-2">
                                <!-- Probability Meter Badge -->
                                <span class="px-3 py-1 rounded-full text-xs font-black flex items-center gap-1.5"
                                    :class="q.repeat_probability >= 0.85 ? 'bg-amber-100 text-amber-900 border border-amber-300' : 'bg-blue-100 text-blue-900 border border-blue-200'">
                                    <i class="fa-solid fa-fire text-amber-500"></i>
                                    रिपीट संभावना: <strong x-text="Math.round(q.repeat_probability * 100) + '%'"></strong>
                                </span>
                                <span class="px-2.5 py-1 rounded-full text-xs font-bold bg-purple-100 text-purple-800" x-text="q.repetition_count + ' बार पूछा गया'"></span>
                            </div>
                        </div>

                        <!-- Question Bilingual -->
                        <div class="mb-4 space-y-1">
                            <h4 class="text-base font-bold text-slate-900 leading-snug" x-text="q.question_en"></h4>
                            <p class="text-sm font-medium text-slate-600" x-show="q.question_hi" x-text="q.question_hi"></p>
                        </div>

                        <!-- 4 Options Grid with Correct Marked in Green -->
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-2.5 mb-4">
                            <div class="p-3 rounded-xl border text-xs flex items-center gap-2" :class="q.correct_option === 1 ? 'bg-emerald-50 border-emerald-400 text-emerald-950 font-bold shadow-sm' : 'bg-slate-50 border-slate-200 text-slate-700'">
                                <span class="w-5 h-5 rounded-full flex items-center justify-center font-bold text-[10px]" :class="q.correct_option === 1 ? 'bg-emerald-600 text-white' : 'bg-slate-200'">1</span>
                                <div><span x-text="q.opt1_en"></span> <span class="text-slate-500 block text-[11px]" x-show="q.opt1_hi" x-text="q.opt1_hi"></span></div>
                            </div>
                            <div class="p-3 rounded-xl border text-xs flex items-center gap-2" :class="q.correct_option === 2 ? 'bg-emerald-50 border-emerald-400 text-emerald-950 font-bold shadow-sm' : 'bg-slate-50 border-slate-200 text-slate-700'">
                                <span class="w-5 h-5 rounded-full flex items-center justify-center font-bold text-[10px]" :class="q.correct_option === 2 ? 'bg-emerald-600 text-white' : 'bg-slate-200'">2</span>
                                <div><span x-text="q.opt2_en"></span> <span class="text-slate-500 block text-[11px]" x-show="q.opt2_hi" x-text="q.opt2_hi"></span></div>
                            </div>
                            <div class="p-3 rounded-xl border text-xs flex items-center gap-2" :class="q.correct_option === 3 ? 'bg-emerald-50 border-emerald-400 text-emerald-950 font-bold shadow-sm' : 'bg-slate-50 border-slate-200 text-slate-700'">
                                <span class="w-5 h-5 rounded-full flex items-center justify-center font-bold text-[10px]" :class="q.correct_option === 3 ? 'bg-emerald-600 text-white' : 'bg-slate-200'">3</span>
                                <div><span x-text="q.opt3_en"></span> <span class="text-slate-500 block text-[11px]" x-show="q.opt3_hi" x-text="q.opt3_hi"></span></div>
                            </div>
                            <div class="p-3 rounded-xl border text-xs flex items-center gap-2" :class="q.correct_option === 4 ? 'bg-emerald-50 border-emerald-400 text-emerald-950 font-bold shadow-sm' : 'bg-slate-50 border-slate-200 text-slate-700'">
                                <span class="w-5 h-5 rounded-full flex items-center justify-center font-bold text-[10px]" :class="q.correct_option === 4 ? 'bg-emerald-600 text-white' : 'bg-slate-200'">4</span>
                                <div><span x-text="q.opt4_en"></span> <span class="text-slate-500 block text-[11px]" x-show="q.opt4_hi" x-text="q.opt4_hi"></span></div>
                            </div>
                        </div>

                        <!-- Card Action: Open in Trainer with AI explanation -->
                        <div class="flex justify-end gap-2 border-t border-slate-100 pt-3">
                            <button @click="testThisQuestion(q)" class="px-4 py-1.5 bg-blue-50 hover:bg-blue-100 text-blue-700 rounded-lg text-xs font-bold transition flex items-center gap-1.5">
                                <i class="fa-solid fa-play text-blue-600"></i> इस प्रश्न का टेस्ट दें
                            </button>
                        </div>
                    </div>
                </template>
            </div>
        </section>

        <!-- ======================================================== -->
        <!-- TAB 3: AI PERSONAL TRAINER & QUIZ (एआई पर्सनल ट्रेनर) -->
        <!-- ======================================================== -->
        <section x-show="activeTab === 'quiz'" x-cloak class="max-w-3xl mx-auto space-y-6">
            <!-- Training Controls Header -->
            <div class="glass-panel p-4 sm:p-5 rounded-3xl shadow-sm border border-slate-200 flex flex-wrap items-center justify-between gap-3">
                <!-- Mode Selector -->
                <div class="flex items-center gap-2">
                    <span class="text-xs font-extrabold text-slate-600 uppercase">मोड:</span>
                    <select x-model="quizMode" @change="loadNextQuestion()" class="bg-slate-100 border border-slate-300 rounded-xl px-3 py-1.5 text-xs font-bold text-slate-800 outline-none focus:ring-2 focus:ring-blue-500">
                        <option value="adaptive">🎯 एडेप्टिव वीकनेस ड्रिल (Weakness Focus)</option>
                        <option value="high_yield">🔥 हाई-यील्ड रिपीट प्रश्न (Repeat Only)</option>
                        <option value="random">🎲 रैंडम प्रश्न (Mix Practice)</option>
                    </select>
                </div>

                <!-- Language Toggle & Speech Aloud -->
                <div class="flex items-center gap-2">
                    <button @click="readQuestionAloud()" class="p-2 bg-slate-100 hover:bg-blue-100 text-slate-700 hover:text-blue-700 rounded-xl text-xs font-bold transition flex items-center gap-1.5" title="प्रश्न को बोलकर सुनें">
                        <i class="fa-solid fa-volume-high"></i>
                        <span class="hidden sm:inline">बोलकर सुनें</span>
                    </button>

                    <div class="flex bg-slate-100 p-1 rounded-xl text-xs font-bold">
                        <button @click="langView = 'both'" :class="langView === 'both' ? 'bg-white shadow-sm text-blue-700' : 'text-slate-600'" class="px-2.5 py-1 rounded-lg transition">दोनों</button>
                        <button @click="langView = 'en'" :class="langView === 'en' ? 'bg-white shadow-sm text-blue-700' : 'text-slate-600'" class="px-2.5 py-1 rounded-lg transition">English</button>
                        <button @click="langView = 'hi'" :class="langView === 'hi' ? 'bg-white shadow-sm text-blue-700' : 'text-slate-600'" class="px-2.5 py-1 rounded-lg transition">हिन्दी</button>
                    </div>
                </div>
            </div>

            <!-- Active Question Card -->
            <div class="glass-panel p-6 sm:p-8 rounded-3xl shadow-md border border-slate-200 relative overflow-hidden" x-show="currentQuestion">
                
                <!-- Badge row -->
                <div class="flex flex-wrap items-center justify-between gap-2 mb-4">
                    <div class="flex items-center gap-2">
                        <span class="bg-blue-600 text-white text-xs font-extrabold px-3.5 py-1 rounded-xl" x-text="currentQuestion?.section"></span>
                        <span class="text-xs font-bold text-slate-400" x-text="'प्रश्न #' + currentQuestion?.question_number"></span>
                    </div>

                    <template x-if="currentQuestion?.repeat_probability">
                        <span class="bg-amber-100 text-amber-900 border border-amber-300 text-xs font-black px-3 py-1 rounded-full flex items-center gap-1.5">
                            <i class="fa-solid fa-fire text-amber-500"></i> रिपीट सम्भावना: <span x-text="Math.round(currentQuestion.repeat_probability * 100) + '%'"></span>
                        </span>
                    </template>
                </div>

                <!-- Question Statement -->
                <div class="mb-6 space-y-2">
                    <h3 class="text-lg sm:text-xl font-bold text-slate-900 leading-snug"
                        x-show="langView === 'both' || langView === 'en'"
                        x-text="currentQuestion?.question_en"></h3>
                    <p class="text-base text-slate-700 font-medium leading-relaxed"
                        x-show="(langView === 'both' || langView === 'hi') && currentQuestion?.question_hi"
                        x-text="currentQuestion?.question_hi"></p>
                </div>

                <!-- 4 Interactive Option Buttons -->
                <div class="space-y-3 mb-6">
                    <template x-for="optNum in [1, 2, 3, 4]" :key="optNum">
                        <button @click="selectOption(optNum)" :disabled="answered"
                            class="w-full text-left p-4 rounded-2xl border-2 transition flex items-start gap-3.5"
                            :class="getOptionClasses(optNum)">
                            <span class="w-7 h-7 rounded-xl flex items-center justify-center text-xs font-black shrink-0 mt-0.5"
                                :class="selectedOption === optNum ? 'bg-blue-600 text-white' : 'bg-slate-200 text-slate-800'"
                                x-text="optNum"></span>
                            <div class="text-sm">
                                <p class="font-bold text-slate-800" x-show="langView === 'both' || langView === 'en'" x-text="currentQuestion?.['opt' + optNum + '_en']"></p>
                                <p class="text-xs text-slate-600 mt-0.5 font-medium" x-show="(langView === 'both' || langView === 'hi') && currentQuestion?.['opt' + optNum + '_hi']" x-text="currentQuestion?.['opt' + optNum + '_hi']"></p>
                            </div>
                        </button>
                    </template>
                </div>

                <!-- Buttons: Submit / Next Question -->
                <div class="flex items-center justify-between pt-2 border-t border-slate-100">
                    <span class="text-xs text-slate-400 font-medium">कीबोर्ड पर 1, 2, 3 या 4 दबाकर भी उत्तर दे सकते हैं।</span>
                    <div class="flex gap-2">
                        <button @click="submitAnswer()" x-show="!answered" :disabled="!selectedOption || loading"
                            class="px-6 py-2.5 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white rounded-xl font-bold text-sm shadow-md transition flex items-center gap-2">
                            <i class="fa-solid fa-check"></i> उत्तर सबमिट करें
                        </button>
                        <button @click="loadNextQuestion()" x-show="answered"
                            class="px-6 py-2.5 bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl font-bold text-sm shadow-md transition flex items-center gap-2">
                            अगला प्रश्न &rarr;
                        </button>
                    </div>
                </div>

                <!-- ==================== HINGLISH AI EXPLANATION BOX ==================== -->
                <div x-show="answered && explanation" class="mt-6 p-5 sm:p-6 bg-gradient-to-br from-amber-50 via-orange-50 to-amber-100/60 border border-amber-300 rounded-3xl shadow-sm">
                    <div class="flex items-center justify-between mb-3 border-b border-amber-200/80 pb-2">
                        <h4 class="font-extrabold text-amber-950 flex items-center gap-2 text-sm sm:text-base">
                            <i class="fa-solid fa-brain text-amber-600"></i> AI कोच हिंग्लिश व्याख्या (Explanation)
                        </h4>
                        <span class="text-xs font-bold text-amber-800 bg-amber-200/70 px-2.5 py-0.5 rounded-full">Gemini 2.5 Flash</span>
                    </div>

                    <div class="prose prose-sm text-slate-900 whitespace-pre-line text-sm leading-relaxed font-medium" x-text="explanation"></div>

                    <!-- Ask Doubt / Guru Ji Se Pucho Mini Chat -->
                    <div class="mt-5 pt-4 border-t border-amber-200/80">
                        <p class="text-xs font-bold text-amber-950 mb-2 flex items-center gap-1.5">
                            <i class="fa-solid fa-comments text-amber-700"></i> कोई संदेह है? AI गुरु से सीधे पूछें (Ask a Doubt):
                        </p>
                        <div class="flex gap-2">
                            <input type="text" x-model="doubtInput" @keydown.enter="askDoubt()" placeholder="उदा: यह विकल्प गलत क्यों है? या कोई शॉर्टकट ट्रिक बताएं..."
                                class="flex-1 bg-white border border-amber-300 rounded-xl px-3.5 py-2 text-xs font-medium outline-none focus:ring-2 focus:ring-amber-500">
                            <button @click="askDoubt()" :disabled="!doubtInput.trim() || doubtLoading" class="px-4 py-2 bg-amber-600 hover:bg-amber-700 text-white rounded-xl text-xs font-bold transition flex items-center gap-1.5 shrink-0">
                                <i class="fa-solid fa-paper-plane" :class="doubtLoading ? 'fa-spin' : ''"></i> पूछें
                            </button>
                        </div>
                        <!-- Doubt answer box -->
                        <div x-show="doubtAnswer" class="mt-3 p-3.5 bg-white/90 rounded-2xl border border-amber-300 text-xs text-slate-800 leading-relaxed font-medium">
                            <strong class="text-amber-900 block mb-1">🤖 AI कोच का जवाब:</strong>
                            <p class="whitespace-pre-line" x-text="doubtAnswer"></p>
                        </div>
                    </div>
                </div>

            </div>
        </section>

        <!-- ======================================================== -->
        <!-- TAB 4: OFFICIAL MOCK TEST SIMULATOR (75 QUESTIONS) -->
        <!-- ======================================================== -->
        <section x-show="activeTab === 'mock_exam'" x-cloak class="space-y-6">
            <!-- Mock Exam Pre-start or Active View -->
            <div x-show="!mockActive" class="glass-panel p-8 rounded-3xl shadow-sm border border-slate-200 text-center max-w-2xl mx-auto space-y-4">
                <div class="w-16 h-16 bg-blue-100 text-blue-600 rounded-3xl flex items-center justify-center text-3xl mx-auto">
                    <i class="fa-solid fa-laptop-file"></i>
                </div>
                <h2 class="text-2xl font-black text-slate-800">MP CPCT आधिकारिक मॉक टेस्ट सिम्युलेटर</h2>
                <p class="text-slate-600 text-sm leading-relaxed">
                    यह टेस्ट बिल्कुल असली CPCT परीक्षा की तरह <strong>75 प्रश्नों</strong> (52 कम्प्यूटर + 5 RC + 6 गणित + 6 रीजनिंग + 6 GK) और <strong>75 मिनट की समय-सीमा</strong> के साथ होता है।
                </p>

                <div class="grid grid-cols-2 sm:grid-cols-3 gap-3 text-left max-w-md mx-auto text-xs py-2">
                    <div class="p-2.5 bg-slate-50 rounded-xl border"><strong class="block text-slate-800">कुल प्रश्न:</strong> 75 MCQs</div>
                    <div class="p-2.5 bg-slate-50 rounded-xl border"><strong class="block text-slate-800">समय:</strong> 75 मिनट</div>
                    <div class="p-2.5 bg-slate-50 rounded-xl border"><strong class="block text-slate-800">पासिंग अंक:</strong> 38 (50%)</div>
                    <div class="p-2.5 bg-slate-50 rounded-xl border"><strong class="block text-slate-800">नेगेटिव मार्किंग:</strong> कोई नहीं (0)</div>
                    <div class="p-2.5 bg-slate-50 rounded-xl border"><strong class="block text-slate-800">भाषा:</strong> English + हिन्दी</div>
                    <div class="p-2.5 bg-slate-50 rounded-xl border"><strong class="block text-slate-800">स्कोरकार्ड:</strong> तुरंत विस्तृत रिपोर्ट</div>
                </div>

                <button @click="startMockExam()" class="px-8 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-xl font-black text-base shadow-lg shadow-blue-500/20 hover:scale-105 active:scale-95 transition">
                    🚀 अभी मॉक टेस्ट शुरू करें
                </button>
            </div>

            <!-- Active Mock Test Interface (TCS/iON Style) -->
            <div x-show="mockActive" class="space-y-4">
                <!-- Test Top Bar: Timer & Section Tabs -->
                <div class="glass-panel p-4 rounded-2xl shadow-sm border border-slate-200 flex flex-wrap items-center justify-between gap-3">
                    <div class="flex items-center gap-2">
                        <span class="px-3 py-1 bg-red-100 text-red-700 font-black rounded-xl text-sm flex items-center gap-1.5">
                            <i class="fa-regular fa-clock"></i> <span x-text="formatTime(mockTimeLeft)">75:00</span>
                        </span>
                        <span class="text-xs font-bold text-slate-500">CPCT Shift Simulation</span>
                    </div>

                    <!-- Jump to Section -->
                    <div class="flex flex-wrap gap-1">
                        <button @click="jumpToSection('Computer Proficiency')" class="px-2.5 py-1 text-xs font-bold rounded-lg bg-blue-100 text-blue-800">कम्प्यूटर (52)</button>
                        <button @click="jumpToSection('Reading Comprehension')" class="px-2.5 py-1 text-xs font-bold rounded-lg bg-slate-100 text-slate-700">RC (5)</button>
                        <button @click="jumpToSection('Quantitative Aptitude')" class="px-2.5 py-1 text-xs font-bold rounded-lg bg-slate-100 text-slate-700">गणित (6)</button>
                        <button @click="jumpToSection('Reasoning Aptitude')" class="px-2.5 py-1 text-xs font-bold rounded-lg bg-slate-100 text-slate-700">रीजनिंग (6)</button>
                        <button @click="jumpToSection('General Awareness')" class="px-2.5 py-1 text-xs font-bold rounded-lg bg-slate-100 text-slate-700">GK (6)</button>
                    </div>

                    <button @click="submitMockExam()" class="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white font-black text-xs rounded-xl shadow transition">
                        टेस्ट समाप्त करें (Submit)
                    </button>
                </div>

                <!-- Main Layout: Question + Question Palette -->
                <div class="grid grid-cols-1 lg:grid-cols-4 gap-4">
                    <!-- Question Canvas (3 Cols) -->
                    <div class="lg:col-span-3 glass-panel p-6 rounded-3xl shadow-sm border border-slate-200 space-y-5">
                        <div class="flex justify-between items-center border-b border-slate-100 pb-3">
                            <span class="text-xs font-bold text-blue-600 uppercase" x-text="mockQuestions[mockCurrentIdx]?.section"></span>
                            <span class="text-xs font-black text-slate-500" x-text="'प्रश्न ' + (mockCurrentIdx + 1) + ' / ' + mockQuestions.length"></span>
                        </div>

                        <!-- Question Texts -->
                        <div class="space-y-2">
                            <h3 class="text-lg font-bold text-slate-900" x-text="mockQuestions[mockCurrentIdx]?.question_en"></h3>
                            <p class="text-sm font-medium text-slate-600" x-text="mockQuestions[mockCurrentIdx]?.question_hi"></p>
                        </div>

                        <!-- 4 Options -->
                        <div class="space-y-2.5">
                            <template x-for="opt in [1, 2, 3, 4]" :key="opt">
                                <label class="flex items-start gap-3 p-3.5 rounded-xl border-2 cursor-pointer transition"
                                    :class="mockAnswers[mockCurrentIdx] === opt ? 'border-blue-600 bg-blue-50/50' : 'border-slate-200 hover:border-slate-300 bg-white'">
                                    <input type="radio" :name="'mock_q_' + mockCurrentIdx" :value="opt"
                                        @change="selectMockOption(opt)"
                                        :checked="mockAnswers[mockCurrentIdx] === opt"
                                        class="mt-1 text-blue-600 focus:ring-blue-500">
                                    <div class="text-xs sm:text-sm">
                                        <p class="font-bold text-slate-800" x-text="mockQuestions[mockCurrentIdx]?.['opt' + opt + '_en']"></p>
                                        <p class="text-xs text-slate-500 mt-0.5" x-text="mockQuestions[mockCurrentIdx]?.['opt' + opt + '_hi']"></p>
                                    </div>
                                </label>
                            </template>
                        </div>

                        <!-- Action Bar: Clear / Prev / Next -->
                        <div class="flex items-center justify-between pt-4 border-t border-slate-100">
                            <button @click="clearMockAnswer()" class="text-xs font-bold text-slate-500 hover:text-red-600">
                                उत्तर हटाएं (Clear)
                            </button>
                            <div class="flex gap-2">
                                <button @click="mockCurrentIdx = Math.max(0, mockCurrentIdx - 1)" :disabled="mockCurrentIdx === 0" class="px-4 py-2 bg-slate-100 text-slate-700 disabled:opacity-40 rounded-xl text-xs font-bold">
                                    &larr; पिछला
                                </button>
                                <button @click="mockCurrentIdx = Math.min(mockQuestions.length - 1, mockCurrentIdx + 1)" class="px-5 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-xl text-xs font-bold">
                                    सहेजें & अगला (Next) &rarr;
                                </button>
                            </div>
                        </div>
                    </div>

                    <!-- Question Palette Grid (1 Col) -->
                    <div class="glass-panel p-4 rounded-3xl shadow-sm border border-slate-200 space-y-4">
                        <h4 class="text-xs font-black uppercase text-slate-700">प्रश्न पैलेट (1 to 75)</h4>
                        <div class="grid grid-cols-5 gap-1.5 max-h-96 overflow-y-auto pr-1">
                            <template x-for="(q, idx) in mockQuestions" :key="idx">
                                <button @click="mockCurrentIdx = idx"
                                    class="w-full aspect-square rounded-lg text-xs font-bold flex items-center justify-center transition border"
                                    :class="getPaletteClass(idx)"
                                    x-text="idx + 1">
                                </button>
                            </template>
                        </div>

                        <!-- Legend -->
                        <div class="text-[11px] space-y-1 text-slate-600 border-t border-slate-100 pt-3">
                            <div class="flex items-center gap-1.5"><span class="w-3 h-3 bg-emerald-500 rounded-sm"></span> उत्तर दिया (Answered)</div>
                            <div class="flex items-center gap-1.5"><span class="w-3 h-3 bg-slate-200 rounded-sm"></span> नहीं देखा (Not Visited)</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Mock Exam Result Scorecard Modal -->
            <div x-show="mockSubmitted" x-cloak class="fixed inset-0 z-50 bg-slate-900/60 backdrop-blur-sm flex items-center justify-center p-4">
                <div class="bg-white rounded-3xl max-w-lg w-full p-6 sm:p-8 shadow-2xl text-center space-y-4 max-h-[90vh] overflow-y-auto">
                    <div class="w-16 h-16 rounded-3xl mx-auto flex items-center justify-center text-3xl font-black"
                        :class="mockScore >= 38 ? 'bg-emerald-100 text-emerald-600' : 'bg-rose-100 text-rose-600'">
                        <i :class="mockScore >= 38 ? 'fa-solid fa-trophy' : 'fa-solid fa-triangle-exclamation'"></i>
                    </div>

                    <h3 class="text-2xl font-black text-slate-800" x-text="mockScore >= 38 ? '🎉 बधाई हो! आप CPCT पास हो गए!' : '⚠️ थोड़ा और अभ्यास जरूरी है!'"></h3>
                    <p class="text-xs text-slate-500">MP CPCT क्वालीफाइंग कटऑफ: 38 अंक (50%)</p>

                    <!-- Score Card Grid -->
                    <div class="grid grid-cols-3 gap-2.5 p-4 bg-slate-50 rounded-2xl border text-xs">
                        <div>
                            <span class="text-slate-500 font-bold block">प्राप्तांक (Score)</span>
                            <span class="text-xl font-black text-blue-600" x-text="mockScore + ' / 75'"></span>
                        </div>
                        <div>
                            <span class="text-slate-500 font-bold block">सटीकता (Accuracy)</span>
                            <span class="text-xl font-black text-emerald-600" x-text="Math.round((mockScore / 75) * 100) + '%'"></span>
                        </div>
                        <div>
                            <span class="text-slate-500 font-bold block">स्थिति (Status)</span>
                            <span class="text-xl font-black" :class="mockScore >= 38 ? 'text-emerald-600' : 'text-rose-600'" x-text="mockScore >= 38 ? 'PASS' : 'FAIL'"></span>
                        </div>
                    </div>

                    <button @click="mockSubmitted = false; mockActive = false; switchTab('quiz')" class="w-full py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-xl font-bold text-sm shadow transition">
                        वापस ट्रेनर पर जाएं
                    </button>
                </div>
            </div>
        </section>

        <!-- ======================================================== -->
        <!-- TAB 5: SEMANTIC CONCEPT SEARCH (सिमेंटिक वेक्टर सर्च) -->
        <!-- ======================================================== -->
        <section x-show="activeTab === 'search'" x-cloak class="max-w-3xl mx-auto space-y-6">
            <div class="glass-panel p-6 sm:p-8 rounded-3xl shadow-sm border border-slate-200 text-center space-y-3">
                <h2 class="text-2xl font-black text-slate-800">स्मार्ट सिमेंटिक कॉन्सेप्ट सर्च</h2>
                <p class="text-slate-500 text-xs sm:text-sm">ChromaDB वेक्टर एम्बेडिंग्स द्वारा किसी भी कॉन्सेप्ट, शॉर्टकट या टॉपिक को स्वाभाविक भाषा में खोजें।</p>

                <!-- Search Input Bar -->
                <div class="flex gap-2 p-2 bg-slate-100 rounded-2xl border border-slate-300">
                    <input type="text" x-model="searchQuery" @keydown.enter="runSearch()" placeholder="जैसे: 'role of compiler in machine code', 'print shortcut', 'RAM ROM difference'..."
                        class="flex-1 bg-transparent px-4 py-2.5 text-xs sm:text-sm outline-none font-semibold text-slate-800">
                    <button @click="runSearch()" class="px-5 py-2.5 bg-blue-600 hover:bg-blue-700 text-white rounded-xl text-xs font-bold transition flex items-center gap-1.5 shrink-0">
                        <i class="fa-solid fa-magnifying-glass"></i> खोजें
                    </button>
                </div>

                <!-- Quick Topic Chips -->
                <div class="flex flex-wrap items-center justify-center gap-1.5 pt-2">
                    <span class="text-xs text-slate-400 font-bold">लोकप्रिय विषय:</span>
                    <button @click="searchQuickTopic('compiler interpreter machine code')" class="px-2.5 py-1 bg-white hover:bg-blue-50 border border-slate-200 text-slate-700 rounded-lg text-xs font-medium">Compiler vs Interpreter</button>
                    <button @click="searchQuickTopic('keyboard shortcut key MS Word')" class="px-2.5 py-1 bg-white hover:bg-blue-50 border border-slate-200 text-slate-700 rounded-lg text-xs font-medium">Keyboard Shortcuts</button>
                    <button @click="searchQuickTopic('RAM ROM cache memory')" class="px-2.5 py-1 bg-white hover:bg-blue-50 border border-slate-200 text-slate-700 rounded-lg text-xs font-medium">Memory & Cache</button>
                    <button @click="searchQuickTopic('open source software linux')" class="px-2.5 py-1 bg-white hover:bg-blue-50 border border-slate-200 text-slate-700 rounded-lg text-xs font-medium">Open Source</button>
                    <button @click="searchQuickTopic('graphics card GPU monitor')" class="px-2.5 py-1 bg-white hover:bg-blue-50 border border-slate-200 text-slate-700 rounded-lg text-xs font-medium">Graphics & Hardware</button>
                </div>
            </div>

            <!-- Search Results Display -->
            <div class="space-y-4">
                <template x-for="res in searchResults" :key="res.id">
                    <div class="glass-panel p-5 rounded-3xl shadow-sm border border-slate-200/90 space-y-3">
                        <div class="flex items-center justify-between">
                            <span class="bg-blue-100 text-blue-800 text-xs font-extrabold px-3 py-1 rounded-lg" x-text="res.section"></span>
                            <span class="text-xs font-black text-emerald-600 flex items-center gap-1">
                                <i class="fa-solid fa-bullseye"></i> सिमिलैरिटी मैच: <span x-text="Math.round(res.similarity_score * 100) + '%'"></span>
                            </span>
                        </div>
                        <h4 class="font-bold text-slate-900 text-sm leading-snug" x-text="res.question_en"></h4>
                        <p class="text-xs text-slate-600 font-medium" x-show="res.question_hi" x-text="res.question_hi"></p>
                        <div class="p-3 bg-emerald-50 text-emerald-950 rounded-xl text-xs font-bold border border-emerald-300">
                            🎯 सही उत्तर: Option <span x-text="res.correct_option"></span> - <span x-text="res['opt' + res.correct_option + '_en']"></span>
                        </div>
                    </div>
                </template>
            </div>
        </section>

        <!-- ======================================================== -->
        <!-- TAB 6: CONTROL CENTER & SCRAPER (कंट्रोल सेंटर) -->
        <!-- ======================================================== -->
        <section x-show="activeTab === 'control'" x-cloak class="space-y-6">
            <!-- Action Bar Cards -->
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <!-- 1-Click Scraper -->
                <div class="glass-panel p-5 rounded-3xl shadow-sm border border-slate-200 space-y-3">
                    <div class="w-10 h-10 rounded-xl bg-blue-100 text-blue-600 flex items-center justify-center font-bold text-lg">
                        <i class="fa-solid fa-cloud-arrow-down"></i>
                    </div>
                    <h4 class="font-bold text-slate-900 text-sm">1-Click Auto Scraper</h4>
                    <p class="text-xs text-slate-500">MP CPCT पोर्टल से नए पेपर्स सीधे डाउनलोड और अनज़िप करें।</p>
                    <button @click="downloadLatestPapers(2)" :disabled="actionLoading" class="w-full py-2.5 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white rounded-xl text-xs font-bold transition flex items-center justify-center gap-2">
                        <i class="fa-solid fa-download"></i> 2 नए पेपर्स डाउनलोड करें
                    </button>
                </div>

                <!-- Batch Parser -->
                <div class="glass-panel p-5 rounded-3xl shadow-sm border border-slate-200 space-y-3">
                    <div class="w-10 h-10 rounded-xl bg-indigo-100 text-indigo-600 flex items-center justify-center font-bold text-lg">
                        <i class="fa-solid fa-gears"></i>
                    </div>
                    <h4 class="font-bold text-slate-900 text-sm">AI Batch Parser & Sync</h4>
                    <p class="text-xs text-slate-500">सभी PDF पेपर्स को SQLite & ChromaDB में पुनः पार्स और सिंक करें।</p>
                    <button @click="runFullParserSync()" :disabled="actionLoading" class="w-full py-2.5 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white rounded-xl text-xs font-bold transition flex items-center justify-center gap-2">
                        <i class="fa-solid fa-sync" :class="actionLoading ? 'fa-spin' : ''"></i> फुल पार्स & वेक्टराइज़
                    </button>
                </div>

                <!-- Export & Reset -->
                <div class="glass-panel p-5 rounded-3xl shadow-sm border border-slate-200 space-y-3">
                    <div class="w-10 h-10 rounded-xl bg-emerald-100 text-emerald-600 flex items-center justify-center font-bold text-lg">
                        <i class="fa-solid fa-file-export"></i>
                    </div>
                    <h4 class="font-bold text-slate-900 text-sm">डेटा बैकअप & रीसेट</h4>
                    <p class="text-xs text-slate-500">प्रश्न बैंक को JSON में एक्सपोर्ट करें या अपना स्कोर रीसेट करें।</p>
                    <div class="flex gap-2">
                        <a href="/api/export/questions" download="cpct_questions.json" class="flex-1 py-2.5 bg-slate-100 hover:bg-slate-200 text-slate-800 text-center rounded-xl text-xs font-bold transition">
                            <i class="fa-solid fa-download"></i> JSON
                        </a>
                        <button @click="resetScorecard()" class="flex-1 py-2.5 bg-rose-50 hover:bg-rose-100 text-rose-700 rounded-xl text-xs font-bold transition">
                            स्कोर रीसेट
                        </button>
                    </div>
                </div>
            </div>

            <!-- Live Portal Exam Cycle List -->
            <div class="glass-panel p-6 rounded-3xl shadow-sm border border-slate-200 space-y-4">
                <div class="flex items-center justify-between">
                    <div>
                        <h3 class="text-base font-bold text-slate-800">आधिकारिक CPCT पोर्टल पेपर्स (131 सेशन्स उपलब्ध)</h3>
                        <p class="text-xs text-slate-500">`cpct.mp.gov.in` से लाइव लिंक - किसी भी पेपर को एक क्लिक में डाउनलोड व पार्स करें।</p>
                    </div>
                    <button @click="fetchAvailablePapers()" class="px-3.5 py-1.5 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-xl text-xs font-bold transition flex items-center gap-1.5">
                        <i class="fa-solid fa-arrows-rotate"></i> रिफ्रेश लिस्ट
                    </button>
                </div>

                <div class="overflow-x-auto max-h-96">
                    <table class="w-full text-left text-sm">
                        <thead class="bg-slate-50 text-slate-600 font-bold text-xs uppercase sticky top-0 border-b">
                            <tr>
                                <th class="py-2.5 px-3">परीक्षा तिथि</th>
                                <th class="py-2.5 px-3">प्रश्न पत्र विवरण</th>
                                <th class="py-2.5 px-3 text-right">डाउनलोड</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-slate-100 text-xs">
                            <template x-for="paper in availablePapers" :key="paper.exam_date">
                                <tr class="hover:bg-slate-50 transition">
                                    <td class="py-2.5 px-3 font-bold text-slate-800" x-text="paper.exam_date"></td>
                                    <td class="py-2.5 px-3 text-slate-600" x-text="paper.title"></td>
                                    <td class="py-2.5 px-3 text-right">
                                        <button @click="downloadSinglePaper(paper.exam_date)" class="px-3 py-1 bg-blue-50 hover:bg-blue-600 hover:text-white text-blue-700 rounded-lg font-bold transition">
                                            डाउनलोड & पार्स
                                        </button>
                                    </td>
                                </tr>
                            </template>
                        </tbody>
                    </table>
                </div>
            </div>
        </section>

    </main>

    <!-- ==================== CLIENT LOGIC (Alpine.js) ==================== -->
    <script>
        function cpctApp() {
            return {
                activeTab: 'dashboard',
                loading: false,
                actionLoading: false,
                soundEnabled: true,
                status: {},
                highYieldQuestions: [],
                filterSection: '',
                currentQuestion: null,
                selectedOption: null,
                answered: false,
                explanation: '',
                quizMode: 'adaptive',
                langView: 'both',
                searchQuery: '',
                searchResults: [],
                availablePapers: [],
                doubtInput: '',
                doubtAnswer: '',
                doubtLoading: false,

                // Mock Test States
                mockActive: false,
                mockQuestions: [],
                mockAnswers: {},
                mockCurrentIdx: 0,
                mockTimeLeft: 4500, // 75 mins in seconds
                mockTimerInterval: null,
                mockSubmitted: false,
                mockScore: 0,

                async initApp() {
                    await this.refreshStatus();
                    await this.loadHighYield();
                    await this.loadNextQuestion();
                    await this.fetchAvailablePapers();

                    // Keyboard shortcuts for answering 1, 2, 3, 4
                    window.addEventListener('keydown', (e) => {
                        if (this.activeTab === 'quiz' && !this.answered && ['1','2','3','4'].includes(e.key)) {
                            this.selectOption(parseInt(e.key));
                        }
                    });
                },

                switchTab(tab) {
                    this.activeTab = tab;
                    window.scrollTo({ top: 0, behavior: 'smooth' });
                },

                toggleSound() {
                    this.soundEnabled = !this.soundEnabled;
                },

                // Web Audio API Synthesizer (Zero external audio files needed!)
                playSound(type) {
                    if (!this.soundEnabled) return;
                    try {
                        const ctx = new (window.AudioContext || window.webkitAudioContext)();
                        const osc = ctx.createOscillator();
                        const gain = ctx.createGain();
                        osc.connect(gain);
                        gain.connect(ctx.destination);

                        if (type === 'correct') {
                            // Pleasant two-tone chime
                            osc.type = 'sine';
                            osc.frequency.setValueAtTime(523.25, ctx.currentTime); // C5
                            osc.frequency.setValueAtTime(659.25, ctx.currentTime + 0.12); // E5
                            gain.gain.setValueAtTime(0.15, ctx.currentTime);
                            gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.35);
                            osc.start();
                            osc.stop(ctx.currentTime + 0.35);
                        } else {
                            // Soft error alert
                            osc.type = 'triangle';
                            osc.frequency.setValueAtTime(220, ctx.currentTime);
                            gain.gain.setValueAtTime(0.12, ctx.currentTime);
                            gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.25);
                            osc.start();
                            osc.stop(ctx.currentTime + 0.25);
                        }
                    } catch (e) {
                        console.warn('Audio synthesis not permitted without interaction:', e);
                    }
                },

                // Text-to-Speech (Reads question aloud in natural voice)
                readQuestionAloud() {
                    if (!('speechSynthesis' in window) || !this.currentQuestion) return;
                    window.speechSynthesis.cancel();
                    const text = this.langView === 'hi' 
                        ? (this.currentQuestion.question_hi || this.currentQuestion.question_en)
                        : this.currentQuestion.question_en;
                    const utterance = new SpeechSynthesisUtterance(text);
                    utterance.lang = this.langView === 'hi' ? 'hi-IN' : 'en-IN';
                    utterance.rate = 0.95;
                    window.speechSynthesis.speak(utterance);
                },

                async refreshStatus() {
                    this.loading = true;
                    try {
                        const res = await fetch('/api/status');
                        this.status = await res.json();
                    } catch (e) {
                        console.error('Status fetch error:', e);
                    } finally {
                        this.loading = false;
                    }
                },

                async loadHighYield() {
                    let url = '/api/trends/high-yield?limit=30';
                    if (this.filterSection) url += `&section=${encodeURIComponent(this.filterSection)}`;
                    try {
                        const res = await fetch(url);
                        const data = await res.json();
                        this.highYieldQuestions = data.questions || [];
                    } catch (e) {
                        console.error('High-yield fetch error:', e);
                    }
                },

                async loadNextQuestion() {
                    this.answered = false;
                    this.selectedOption = null;
                    this.explanation = '';
                    this.doubtInput = '';
                    this.doubtAnswer = '';
                    try {
                        const res = await fetch(`/api/quiz/next?mode=${this.quizMode}`);
                        if (res.ok) {
                            this.currentQuestion = await res.json();
                        }
                    } catch (e) {
                        console.error('Next question error:', e);
                    }
                },

                startAdaptiveSectionDrill(sectionName) {
                    this.quizMode = 'adaptive';
                    this.switchTab('quiz');
                    this.loadNextQuestion();
                },

                testThisQuestion(q) {
                    this.currentQuestion = q;
                    this.answered = false;
                    this.selectedOption = null;
                    this.explanation = '';
                    this.switchTab('quiz');
                },

                selectOption(num) {
                    if (this.answered) return;
                    this.selectedOption = num;
                },

                getOptionClasses(num) {
                    if (!this.answered) {
                        return this.selectedOption === num 
                            ? 'border-blue-600 bg-blue-50/70 shadow-sm' 
                            : 'border-slate-200 hover:border-slate-300 bg-white';
                    }
                    if (num === this.currentQuestion.correct_option) {
                        return 'border-emerald-500 bg-emerald-50 text-emerald-950 font-bold';
                    }
                    if (num === this.selectedOption && num !== this.currentQuestion.correct_option) {
                        return 'border-rose-500 bg-rose-50 text-rose-950';
                    }
                    return 'border-slate-200 opacity-60 bg-white';
                },

                async submitAnswer() {
                    if (!this.selectedOption || this.answered) return;
                    this.loading = true;
                    try {
                        const res = await fetch('/api/quiz/submit', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                question_id: this.currentQuestion.id,
                                selected_option: this.selectedOption,
                                time_spent_seconds: 5.0
                            })
                        });
                        const data = await res.json();
                        this.answered = true;
                        this.explanation = data.explanation;

                        if (data.is_correct) {
                            this.playSound('correct');
                        } else {
                            this.playSound('incorrect');
                        }
                        await this.refreshStatus();
                    } catch (e) {
                        alert('Submission error: ' + e);
                    } finally {
                        this.loading = false;
                    }
                },

                async askDoubt() {
                    if (!this.doubtInput.trim() || !this.currentQuestion) return;
                    this.doubtLoading = true;
                    try {
                        const res = await fetch('/api/chat/ask', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                question_id: this.currentQuestion.id,
                                doubt_query: this.doubtInput
                            })
                        });
                        const data = await res.json();
                        this.doubtAnswer = data.answer;
                    } catch (e) {
                        this.doubtAnswer = "Error connecting to AI Coach: " + e;
                    } finally {
                        this.doubtLoading = false;
                    }
                },

                async runSearch() {
                    if (!this.searchQuery.trim()) return;
                    try {
                        const res = await fetch(`/api/search?query=${encodeURIComponent(this.searchQuery)}&limit=10`);
                        const data = await res.json();
                        this.searchResults = data.results || [];
                    } catch (e) {
                        console.error('Search error:', e);
                    }
                },

                searchQuickTopic(topic) {
                    this.searchQuery = topic;
                    this.runSearch();
                },

                async fetchAvailablePapers() {
                    try {
                        const res = await fetch('/api/papers/available');
                        const data = await res.json();
                        this.availablePapers = data.papers || [];
                    } catch (e) {
                        console.warn('Failed to fetch available papers:', e);
                    }
                },

                async downloadSinglePaper(date) {
                    alert(`डाउनलोड शुरू: ${date}... कृपया प्रतीक्षा करें।`);
                    try {
                        await fetch('/api/scraper/download', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ exam_date: date, limit: 1 })
                        });
                        await fetch('/api/parser/run', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ use_vision: false })
                        });
                        alert('पेपर डाउनलोड व पार्स हो गया!');
                        await this.refreshStatus();
                        await this.loadHighYield();
                    } catch (e) {
                        alert('Download error: ' + e);
                    }
                },

                async downloadLatestPapers(count) {
                    this.actionLoading = true;
                    try {
                        await fetch('/api/scraper/download', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ limit: count })
                        });
                        await fetch('/api/parser/run', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ use_vision: false })
                        });
                        alert(`${count} नए पेपर्स डाउनलोड होकर डेटाबेस में जुड़ गए!`);
                        await this.refreshStatus();
                        await this.loadHighYield();
                    } catch (e) {
                        alert('Error: ' + e);
                    } finally {
                        this.actionLoading = false;
                    }
                },

                async runFullParserSync() {
                    this.actionLoading = true;
                    try {
                        const res = await fetch('/api/parser/run', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ use_vision: false })
                        });
                        const data = await res.json();
                        alert(`सफल! ${data.total_extracted} प्रश्न पार्स और वेक्टराइज़ हो गए।`);
                        await this.refreshStatus();
                        await this.loadHighYield();
                    } catch (e) {
                        alert('Parse error: ' + e);
                    } finally {
                        this.actionLoading = false;
                    }
                },

                async resetScorecard() {
                    if (!confirm('क्या आप अपना स्कोरकार्ड और अभ्यास इतिहास रीसेट करना चाहते हैं?')) return;
                    try {
                        await fetch('/api/trainer/reset', { method: 'POST' });
                        alert('स्कोरकार्ड रीसेट हो गया!');
                        await this.refreshStatus();
                    } catch (e) {
                        alert('Reset error: ' + e);
                    }
                },

                // Mock Exam Functions
                async startMockExam() {
                    try {
                        const res = await fetch('/api/mock-exam');
                        const data = await res.json();
                        this.mockQuestions = data.questions || [];
                        this.mockAnswers = {};
                        this.mockCurrentIdx = 0;
                        this.mockTimeLeft = 4500; // 75 mins
                        this.mockActive = true;
                        this.mockSubmitted = false;

                        if (this.mockTimerInterval) clearInterval(this.mockTimerInterval);
                        this.mockTimerInterval = setInterval(() => {
                            if (this.mockTimeLeft > 0) {
                                this.mockTimeLeft--;
                            } else {
                                this.submitMockExam();
                            }
                        }, 1000);
                    } catch (e) {
                        alert('Failed to load mock exam: ' + e);
                    }
                },

                selectMockOption(opt) {
                    this.mockAnswers[this.mockCurrentIdx] = opt;
                },

                clearMockAnswer() {
                    delete this.mockAnswers[this.mockCurrentIdx];
                },

                jumpToSection(sec) {
                    const idx = this.mockQuestions.findIndex(q => q.section === sec);
                    if (idx !== -1) this.mockCurrentIdx = idx;
                },

                getPaletteClass(idx) {
                    if (idx === this.mockCurrentIdx) return 'bg-blue-600 text-white ring-2 ring-blue-400';
                    if (this.mockAnswers[idx]) return 'bg-emerald-500 text-white';
                    return 'bg-slate-100 text-slate-700 hover:bg-slate-200';
                },

                submitMockExam() {
                    clearInterval(this.mockTimerInterval);
                    let score = 0;
                    this.mockQuestions.forEach((q, idx) => {
                        if (this.mockAnswers[idx] === q.correct_option) {
                            score++;
                        }
                    });
                    this.mockScore = score;
                    this.mockSubmitted = true;
                },

                formatTime(seconds) {
                    const m = Math.floor(seconds / 60);
                    const s = seconds % 60;
                    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
                }
            }
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def serve_web_ui():
    """Serve the modern interactive web application."""
    return HTMLResponse(content=WEB_UI_HTML)


# ==========================================
# CLI APPLICATION DISPATCHER
# ==========================================

def run_cli():
    import argparse
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    console = Console()

    parser_cli = argparse.ArgumentParser(description="MP CPCT AI Exam Agent")
    subparsers = parser_cli.add_subparsers(dest="command", help="Available subcommands")

    # scrape command
    p_scrape = subparsers.add_parser("scrape", help="Download & unzip CPCT papers")
    p_scrape.add_argument("--limit", type=int, default=3, help="Number of papers to download")
    p_scrape.add_argument("--date", type=str, help="Filter for specific exam date")
    p_scrape.add_argument("--all", action="store_true", help="Download all available papers")
    p_scrape.add_argument("--list", action="store_true", help="List papers on portal")

    # parse command
    p_parse = subparsers.add_parser("parse", help="Parse PDFs into SQLite database")
    p_parse.add_argument("--pdf", type=str, help="Path to single PDF")
    p_parse.add_argument("--all", action="store_true", help="Parse all PDFs in data/extracted_pdfs")
    p_parse.add_argument("--vision", action="store_true", help="Force Gemini 2.5 Flash Vision OCR")

    # analyze command
    p_analyze = subparsers.add_parser("analyze", help="Vector clustering & repeat question analysis")
    p_analyze.add_argument("--sync", action="store_true", help="Sync SQLite into ChromaDB")
    p_analyze.add_argument("--threshold", type=float, default=0.82, help="Similarity threshold")
    p_analyze.add_argument("--top", type=int, default=15, help="Top repeat questions")
    p_analyze.add_argument("--search", type=str, help="Semantic search query")

    # train command
    p_train = subparsers.add_parser("train", help="Run interactive personal trainer quiz")
    p_train.add_argument("--mode", type=str, default="adaptive", choices=["adaptive", "high_yield", "mock", "section"])

    # serve command
    p_serve = subparsers.add_parser("serve", help="Run FastAPI web server")
    p_serve.add_argument("--host", type=str, default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)

    # status command
    subparsers.add_parser("status", help="Display overall system status")

    args = parser_cli.parse_args()

    if args.command == "serve":
        console.print(f"[bold green]🚀 Starting MP CPCT AI Agent Web Server on http://{args.host}:{args.port}[/bold green]")
        uvicorn.run(app, host=args.host, port=args.port)
        return

    if args.command == "scrape":
        if args.list:
            papers = scraper.get_available_papers()
            console.print(f"\n[bold]Found {len(papers)} available papers on portal:[/bold]")
            for p in papers[:20]:
                console.print(f" - {p['exam_date']}: {p['qp_url']}")
        else:
            limit = None if args.all else args.limit
            scraper.run_pipeline(limit=limit, target_date=args.date)
        return

    if args.command == "parse":
        if args.pdf:
            qs = parser.parse_pdf(Path(args.pdf), use_vision=args.vision)
            parser.save_to_database(qs, Path(args.pdf))
        else:
            parser.parse_and_store_all(use_vision=args.vision)
            analyzer.sync_to_chroma()
            analyzer.detect_high_yield_questions()
        return

    if args.command == "analyze":
        if args.sync:
            analyzer.sync_to_chroma()
        if args.search:
            res = analyzer.semantic_search(args.search, n_results=args.top)
            console.print(f"\n[bold]Results for '{args.search}':[/bold]")
            for r in res:
                console.print(f"[{r['section']}] (Match: {int(r['similarity_score']*100)}%) {r['question_en']}")
        else:
            analyzer.detect_high_yield_questions(threshold=args.threshold)
            hy = analyzer.get_high_yield_report(limit=args.top)
            table = Table(title="Top High-Yield Repeating Questions", border_style="green")
            table.add_column("No", justify="right")
            table.add_column("Section")
            table.add_column("Repeat Prob")
            table.add_column("Question (EN)")
            for i, q in enumerate(hy, 1):
                prob = f"{int(q['repeat_probability']*100)}%"
                table.add_row(str(i), q["section"], prob, q["question_en"][:70] + "...")
            console.print(table)
        return

    if args.command == "train":
        from cpct_ai_agent.modules.trainer import run_interactive_cli_quiz
        run_interactive_cli_quiz()
        return

    if args.command == "status":
        stats = get_system_status()
        console.print(Panel.fit(
            f"[bold cyan]MP CPCT AI Agent System Status[/bold cyan]\n"
            f"Questions in SQLite: [bold green]{stats['total_questions_in_db']}[/bold green]\n"
            f"ChromaDB Vectors: [bold green]{stats['chroma_embeddings_count']}[/bold green]\n"
            f"Shifts Parsed: [bold green]{stats['total_papers_parsed']}[/bold green]\n"
            f"Gemini 2.5 Flash Configured: {'[bold green]YES[/bold green]' if stats['gemini_configured'] else '[bold yellow]NO (Fallback Active)[/bold yellow]'}",
            border_style="blue"
        ))
        return

    # If no subcommand provided, show welcome menu
    console.print(Panel.fit(
        "[bold cyan]🎯 MP CPCT AI Exam Scraper, Trend Analyzer & Personal Trainer Agent[/bold cyan]\n"
        "[white]Complete pipeline for automated scraping, Gemini Vision parsing, vector clustering & Hinglish training coach.[/white]\n\n"
        "[bold]Available Commands:[/bold]\n"
        "  1. [yellow]python main.py status[/yellow]        - Show system health & database metrics\n"
        "  2. [yellow]python main.py scrape[/yellow]        - Download and extract portal papers\n"
        "  3. [yellow]python main.py parse --all[/yellow]   - Batch parse PDFs to SQLite & ChromaDB\n"
        "  4. [yellow]python main.py analyze[/yellow]       - Run ChromaDB clustering for repeat trends\n"
        "  5. [yellow]python main.py train[/yellow]         - Launch interactive personal training coach\n"
        "  6. [yellow]python main.py serve[/yellow]         - Start FastAPI web server & visual UI\n",
        border_style="green"
    ))


if __name__ == "__main__":
    run_cli()
