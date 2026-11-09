"""
Module 3: ChromaDB Clustering & High-Yield Trend Analyzer
Ingests CPCT question bank into ChromaDB vector store, detects rephrased & repeating
questions across shifts/dates using semantic similarity, and calculates repeat probabilities.
"""

import os
import sys
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import chromadb
from chromadb.config import Settings

# Ensure project root in sys.path
current_file = Path(__file__).resolve()
agent_root = current_file.parent.parent
workspace_root = agent_root.parent
for p in [str(workspace_root), str(agent_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from cpct_ai_agent.config import (
        SQLITE_DB_PATH,
        CHROMA_DIR,
        CHROMA_COLLECTION_NAME,
        SIMILARITY_REPEAT_THRESHOLD,
    )
except ImportError:
    from config import (
        SQLITE_DB_PATH,
        CHROMA_DIR,
        CHROMA_COLLECTION_NAME,
        SIMILARITY_REPEAT_THRESHOLD,
    )


class CPCTAnalyzer:
    """
    Semantic trend analyzer using ChromaDB embeddings to discover
    rephrased and repeating CPCT questions across examination cycles.
    """

    def __init__(
        self,
        db_path: Path = SQLITE_DB_PATH,
        chroma_dir: Path = CHROMA_DIR,
        collection_name: str = CHROMA_COLLECTION_NAME,
        similarity_threshold: float = SIMILARITY_REPEAT_THRESHOLD,
    ):
        self.db_path = Path(db_path)
        self.chroma_dir = Path(chroma_dir)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.similarity_threshold = similarity_threshold

        # Initialize persistent ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.chroma_dir),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._init_trends_table()

    def _get_db_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_trends_table(self):
        """Create question_trends table in SQLite to cache repetition analytics."""
        with self._get_db_conn() as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS question_trends (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id TEXT UNIQUE,
                cluster_id TEXT,
                repetition_count INTEGER DEFAULT 1,
                repeat_probability REAL DEFAULT 0.0,
                topic_tag TEXT,
                is_high_yield INTEGER DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trends_yield ON question_trends(is_high_yield)")
            conn.commit()

    def sync_to_chroma(self, force: bool = False) -> int:
        """
        Sync all questions from SQLite database into ChromaDB vector collection.
        Returns the number of indexed questions.
        """
        with self._get_db_conn() as conn:
            cursor = conn.cursor()
            rows = cursor.execute("""
                SELECT id, question_id, paper_id, question_number, section,
                       question_en, question_hi, correct_option, exam_date, shift
                FROM questions
                WHERE question_en IS NOT NULL AND length(trim(question_en)) > 0
            """).fetchall()

        if not rows:
            print("[!] No questions found in SQLite to sync.")
            return 0

        print(f"[*] Syncing {len(rows)} questions into ChromaDB vector store...")
        documents = []
        metadatas = []
        ids = []

        for row in rows:
            doc_id = str(row["id"])
            # Rich document text combining English, Hindi, and section context
            doc_text = f"[{row['section']}] {row['question_en']}\n{row['question_hi'] or ''}"
            meta = {
                "db_id": row["id"],
                "question_id": str(row["question_id"]),
                "paper_id": row["paper_id"],
                "question_number": row["question_number"],
                "section": row["section"],
                "correct_option": row["correct_option"],
                "exam_date": str(row["exam_date"]),
                "shift": str(row["shift"]),
            }
            documents.append(doc_text)
            metadatas.append(meta)
            ids.append(doc_id)

        # Batch upsert into ChromaDB
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            end_idx = min(i + batch_size, len(ids))
            self.collection.upsert(
                ids=ids[i:end_idx],
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx],
            )

        count = self.collection.count()
        print(f"[OK] ChromaDB collection '{self.collection_name}' now contains {count} embedded questions.")
        return count

    def detect_high_yield_questions(self, threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Perform semantic similarity clustering to detect repeating questions
        across different papers/shifts.
        """
        sim_threshold = threshold or self.similarity_threshold
        total_items = self.collection.count()
        if total_items == 0:
            self.sync_to_chroma()
            total_items = self.collection.count()

        if total_items == 0:
            return []

        print(f"[*] Analyzing semantic clusters across {total_items} questions (Similarity Threshold >= {sim_threshold})...")

        # Fetch all items from ChromaDB
        all_data = self.collection.get(include=["documents", "metadatas"])
        all_ids = all_data["ids"]
        all_docs = all_data["documents"]
        all_metas = all_data["metadatas"]

        clusters: Dict[str, List[str]] = {}
        visited = set()
        repeat_stats: Dict[str, Dict[str, Any]] = {}

        # Query top nearest neighbors for each document
        for i, q_id in enumerate(all_ids):
            query_text = all_docs[i]
            q_meta = all_metas[i]

            results = self.collection.query(
                query_texts=[query_text],
                n_results=min(10, total_items),
                include=["documents", "metadatas", "distances"],
            )

            matched_ids = results["ids"][0]
            matched_distances = results["distances"][0]
            matched_metas = results["metadatas"][0]

            # In cosine space: similarity = 1 - distance
            cluster_members = [q_id]
            for match_id, dist, meta in zip(matched_ids, matched_distances, matched_metas):
                if match_id == q_id:
                    continue
                similarity = 1.0 - dist
                # If similarity meets threshold and comes from different shift or date
                if similarity >= sim_threshold:
                    is_diff_paper = (meta.get("exam_date") != q_meta.get("exam_date") or
                                     meta.get("shift") != q_meta.get("shift"))
                    if is_diff_paper:
                        cluster_members.append(match_id)

            cluster_size = len(set(cluster_members))
            cluster_id = f"cluster_{min(cluster_members)}"

            # Calculate repeat probability score
            # Base probability 50% + 20% for each additional shift it appears in (max 98%)
            repeat_prob = min(0.98, 0.45 + (cluster_size * 0.18)) if cluster_size > 1 else 0.25
            is_high_yield = 1 if cluster_size > 1 else 0

            repeat_stats[q_id] = {
                "cluster_id": cluster_id,
                "repetition_count": cluster_size,
                "repeat_probability": round(repeat_prob, 2),
                "is_high_yield": is_high_yield,
            }

        # Save trends to SQLite
        with self._get_db_conn() as conn:
            cursor = conn.cursor()
            for q_id, stats in repeat_stats.items():
                cursor.execute("""
                INSERT INTO question_trends (question_id, cluster_id, repetition_count, repeat_probability, is_high_yield, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(question_id) DO UPDATE SET
                    cluster_id=excluded.cluster_id,
                    repetition_count=excluded.repetition_count,
                    repeat_probability=excluded.repeat_probability,
                    is_high_yield=excluded.is_high_yield,
                    updated_at=CURRENT_TIMESTAMP
                """, (q_id, stats["cluster_id"], stats["repetition_count"], stats["repeat_probability"], stats["is_high_yield"]))
            conn.commit()

        # Gather top high-yield questions for return
        high_yield = self.get_high_yield_report(limit=30)
        print(f"[OK] Trend Analysis complete: Identified {len([s for s in repeat_stats.values() if s['is_high_yield']])} high-probability repeating questions.")
        return high_yield

    def get_high_yield_report(
        self,
        limit: int = 20,
        section: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch top high-yield questions sorted by repetition frequency and repeat probability.
        """
        with self._get_db_conn() as conn:
            query = """
            SELECT q.id, q.question_number, q.section, q.question_en, q.question_hi,
                   q.opt1_en, q.opt2_en, q.opt3_en, q.opt4_en, q.correct_option,
                   q.exam_date, q.shift,
                   t.repetition_count, t.repeat_probability, t.cluster_id
            FROM questions q
            JOIN question_trends t ON cast(q.id as text) = t.question_id
            WHERE 1=1
            """
            params: List[Any] = []
            if section:
                query += " AND q.section = ?"
                params.append(section)

            query += " ORDER BY t.repeat_probability DESC, t.repetition_count DESC, q.id ASC LIMIT ?"
            params.append(limit)

            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def get_topic_distribution(self) -> List[Dict[str, Any]]:
        """Compute repetition distribution across sections."""
        with self._get_db_conn() as conn:
            rows = conn.execute("""
            SELECT q.section,
                   COUNT(q.id) as total_questions,
                   SUM(CASE WHEN t.is_high_yield = 1 THEN 1 ELSE 0 END) as repeat_count,
                   ROUND(AVG(t.repeat_probability) * 100, 1) as avg_repeat_percentage
            FROM questions q
            LEFT JOIN question_trends t ON cast(q.id as text) = t.question_id
            GROUP BY q.section
            ORDER BY repeat_count DESC
            """).fetchall()
            return [dict(r) for r in rows]

    def semantic_search(
        self,
        query: str,
        n_results: int = 5,
        section: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic vector search to find questions closely matching a concept or query.
        """
        where_filter = {"section": section} if section else None
        res = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        matched_ids = res["ids"][0]
        distances = res["distances"][0]

        if not matched_ids:
            return []

        # Fetch full question details from SQLite
        placeholders = ",".join("?" for _ in matched_ids)
        with self._get_db_conn() as conn:
            cursor = conn.cursor()
            rows = cursor.execute(
                f"""
                SELECT q.*, t.repeat_probability, t.repetition_count
                FROM questions q
                LEFT JOIN question_trends t ON cast(q.id as text) = t.question_id
                WHERE q.id IN ({placeholders})
                """,
                matched_ids,
            ).fetchall()

            row_map = {str(r["id"]): dict(r) for r in rows}
            results = []
            for q_id, dist in zip(matched_ids, distances):
                if q_id in row_map:
                    item = row_map[q_id]
                    item["similarity_score"] = round(1.0 - dist, 4)
                    results.append(item)
            return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MP CPCT Vector Embedding & High-Yield Trend Analyzer")
    parser.add_argument("--sync", action="store_true", help="Sync SQLite questions into ChromaDB")
    parser.add_argument("--analyze", action="store_true", help="Run semantic clustering to detect repeat questions")
    parser.add_argument("--threshold", type=float, default=0.82, help="Cosine similarity threshold for duplicate detection")
    parser.add_argument("--search", type=str, help="Search questions semantically")
    parser.add_argument("--top", type=int, default=10, help="Number of high-yield questions to display")
    parser.add_argument("--section", type=str, help="Filter high-yield by section")
    args = parser.parse_args()

    analyzer = CPCTAnalyzer()

    if args.sync:
        analyzer.sync_to_chroma()

    if args.analyze or (not args.search and not args.sync):
        analyzer.detect_high_yield_questions(threshold=args.threshold)
        high_yield = analyzer.get_high_yield_report(limit=args.top, section=args.section)
        print(f"\nTop {len(high_yield)} High-Probability CPCT Repeating Questions:")
        print("=" * 80)
        for i, q in enumerate(high_yield, 1):
            prob = int(q.get("repeat_probability", 0) * 100)
            print(f"{i:2d}. [{q['section']}] (Repeat Prob: {prob}%, Repeats: {q.get('repetition_count', 1)}x)")
            print(f"    EN: {q['question_en'][:90]}...")
            if q.get('question_hi'):
                print(f"    HI: {q['question_hi'][:90]}...")
            print(f"    Correct Option: {q['correct_option']}")
            print("-" * 80)

        # Print topic breakdown
        dist = analyzer.get_topic_distribution()
        print("\nSection-wise Repeat Distribution:")
        for d in dist:
            print(f" - {d['section']:<30}: {d['repeat_count'] or 0} repeats / {d['total_questions']} total (Avg Prob: {d['avg_repeat_percentage'] or 0}%)")

    if args.search:
        print(f"\n[*] Semantic search results for '{args.search}':")
        results = analyzer.semantic_search(args.search, n_results=args.top, section=args.section)
        for i, r in enumerate(results, 1):
            sim = int(r["similarity_score"] * 100)
            c_opt = r["correct_option"]
            ans_text = r.get(f"opt{c_opt}_en", "")
            print(f"{i:2d}. [Match: {sim}%] [{r['section']}] {r['question_en'][:85]}...")
            print(f"    Ans Option {c_opt}: {ans_text}")



if __name__ == "__main__":
    main()
