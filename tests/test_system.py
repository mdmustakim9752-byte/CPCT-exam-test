"""
End-to-End System & API Verification Test Suite
"""

import sys
import os
from pathlib import Path

# Ensure paths
root_dir = Path(__file__).resolve().parent.parent
agent_dir = root_dir / "cpct_ai_agent"
for p in [str(root_dir), str(agent_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi.testclient import TestClient
from cpct_ai_agent.main import app
from cpct_ai_agent.modules.trainer import CPCTTrainer
from cpct_ai_agent.modules.analyzer import CPCTAnalyzer
from cpct_ai_agent.modules.parser import CPCTParser
from cpct_ai_agent.modules.scraper import CPCTScraper

def test_system():
    client = TestClient(app)
    
    print("[1] Testing GET /api/status...")
    resp = client.get("/api/status")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    data = resp.json()
    print("    -> Status:", data["status"])
    print("    -> Total Questions in DB:", data["total_questions_in_db"])
    print("    -> Chroma Embeddings:", data["chroma_embeddings_count"])
    assert data["total_questions_in_db"] > 0, "No questions in DB"

    print("\n[2] Testing GET / (Web UI)...")
    resp = client.get("/")
    assert resp.status_code == 200
    assert "MP CPCT AI Agent" in resp.text
    print("    -> Web UI rendered successfully (HTML length:", len(resp.text), "bytes)")

    print("\n[3] Testing GET /api/trends/high-yield...")
    resp = client.get("/api/trends/high-yield?limit=5")
    assert resp.status_code == 200
    hy = resp.json()
    print(f"    -> Retrieved {hy['count']} high-yield repeat questions")
    assert len(hy["questions"]) > 0

    print("\n[4] Testing GET /api/quiz/next (Adaptive Mode)...")
    resp = client.get("/api/quiz/next?mode=adaptive")
    assert resp.status_code == 200
    q = resp.json()
    print(f"    -> Next Q: #{q['question_number']} [{q['section']}]")
    print("    -> Text:", q["question_en"][:70] + "...")

    print("\n[5] Testing POST /api/quiz/submit (Evaluation & Hinglish Explanation)...")
    q_id = q["id"]
    submit_resp = client.post("/api/quiz/submit", json={
        "question_id": q_id,
        "selected_option": q["correct_option"],
        "time_spent_seconds": 3.5
    })
    assert submit_resp.status_code == 200
    sub_data = submit_resp.json()
    print("    -> Is Correct:", sub_data["is_correct"])
    print("    -> Explanation Preview:\n       " + sub_data["explanation"][:120] + "...")

    print("\n[6] Testing GET /api/search (Semantic Vector Search)...")
    resp = client.get("/api/search?query=compiler+machine+code&limit=3")
    assert resp.status_code == 200
    s_data = resp.json()
    print(f"    -> Query: '{s_data['query']}', Found: {s_data['count']} matches")
    assert len(s_data["results"]) > 0

    print("\n[7] Testing GET /api/mock-exam (75 Official CPCT MCQs)...")
    resp = client.get("/api/mock-exam")
    assert resp.status_code == 200
    mock_data = resp.json()
    print(f"    -> Generated mock exam with {mock_data['total']} questions across 5 sections")
    assert mock_data["total"] == 75, f"Expected 75 questions, got {mock_data['total']}"

    print("\n" + "=" * 60)
    print("✅ ALL 7 SYSTEM & API TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 60)

if __name__ == "__main__":
    test_system()

