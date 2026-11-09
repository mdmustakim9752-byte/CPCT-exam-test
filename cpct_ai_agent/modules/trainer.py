"""
Module 4: Adaptive Quiz & Hinglish Explanation Engine
Personal AI Training Coach that adapts question difficulty to user weaknesses,
prioritizes high-yield repeating questions, tracks performance metrics in SQLite,
and provides conversational, pedagogical explanations in Hinglish.
"""

import os
import sys
import json
import sqlite3
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

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
        GEMINI_API_KEY,
        GEMINI_MODEL,
        SECTIONS,
        SECTION_WEIGHTS,
    )
except ImportError:
    from config import (
        SQLITE_DB_PATH,
        GEMINI_API_KEY,
        GEMINI_MODEL,
        SECTIONS,
        SECTION_WEIGHTS,
    )


class HinglishExplanationEngine:
    """
    Pedagogical explanation engine that provides clear, memorable concept
    explanations in Hinglish (Hindi + English mix).
    Uses Gemini 2.5 Flash if available, with built-in rule-based fallback.
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
                print(f"[!] Warning: Could not initialize Gemini client: {e}")

    def ask_doubt(self, question: Dict[str, Any], doubt_query: str) -> str:
        """Answer student's follow-up doubt or question in conversational Hinglish using Gemini 2.5 Flash."""
        if not self.client:
            return "Gemini API active nahi hai. Kripya .env me GEMINI_API_KEY check karein."
        prompt = f"""
Act as a super friendly, encouraging MP CPCT Exam Coach & Teacher.
A student is studying this CPCT question and has asked a doubt.

Question (EN): {question.get('question_en', '')}
Question (HI): {question.get('question_hi', '')}
Options:
1. {question.get('opt1_en', '')} | {question.get('opt1_hi', '')}
2. {question.get('opt2_en', '')} | {question.get('opt2_hi', '')}
3. {question.get('opt3_en', '')} | {question.get('opt3_hi', '')}
4. {question.get('opt4_en', '')} | {question.get('opt4_hi', '')}
Correct Option: Option {question.get('correct_option')}

Student's Doubt: "{doubt_query}"

Explain clearly and simply in conversational Hinglish (Hindi + English mix). Use examples, memory shortcuts, or simple analogies so the student completely understands and never forgets in the exam!
"""
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            return f"Error connecting to Gemini: {e}"

    def generate_explanation(self, question: Dict[str, Any], user_choice: Optional[int] = None) -> str:
        """Generate a friendly, structured explanation in Hinglish."""
        if self.client:
            try:
                return self._generate_gemini_explanation(question, user_choice)
            except Exception as e:
                print(f"[!] Gemini explanation failed: {e}. Falling back to template engine.")

        return self._generate_template_explanation(question, user_choice)

    def _generate_gemini_explanation(self, q: Dict[str, Any], user_choice: Optional[int]) -> str:
        prompt = f"""
Act as an Expert MP CPCT Exam Coach & Teacher. Explain this question in engaging, natural Hinglish 
(friendly mix of Hindi and English written in Latin script or Devanagari mix) so that a student can easily remember it for their exam.

Section: {q.get('section', 'Computer Proficiency')}
Question (EN): {q.get('question_en', '')}
Question (HI): {q.get('question_hi', '')}
Option 1: {q.get('opt1_en', '')} | {q.get('opt1_hi', '')}
Option 2: {q.get('opt2_en', '')} | {q.get('opt2_hi', '')}
Option 3: {q.get('opt3_en', '')} | {q.get('opt3_hi', '')}
Option 4: {q.get('opt4_en', '')} | {q.get('opt4_hi', '')}
Correct Option: Option {q.get('correct_option')}
User Selected Option: Option {user_choice if user_choice else 'None'}

Please structure your Hinglish response strictly in these 4 concise sections:
1. 🎯 Sahi Jawab: Clearly state the correct option and what it means.
2. 💡 Aasan Shabdon Mein (Core Concept): Explain the fundamental logic in simple everyday terms.
3. ❌ Baki Options Kyon Galat Hain?: Briefly clarify why the other options don't fit.
4. ⚡ CPCT Exam Hack / Yaad Rakhne Ka Tarika: A mnemonic, shortcut, or quick memory hook.
"""
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return response.text.strip()

    def _generate_template_explanation(self, q: Dict[str, Any], user_choice: Optional[int]) -> str:
        """Deterministic, rich Hinglish explanation when offline or no API key."""
        c_opt = q.get("correct_option", 1)
        c_text_en = q.get(f"opt{c_opt}_en", "")
        c_text_hi = q.get(f"opt{c_opt}_hi", "")
        section = q.get("section", "General")

        is_user_right = (user_choice == c_opt) if user_choice else None
        intro = ""
        if is_user_right is True:
            intro = "🎉 **Bilkul Sahi Jawab! Shabaash!**\n\n"
        elif is_user_right is False:
            intro = f"⚠️ **Aapka chuna gaya option galat tha (Aapne Option {user_choice} chuna).** Ghabraiye mat, concept samajhte hain:\n\n"

        exp = (
            f"{intro}"
            f"🎯 **Sahi Jawab:** Option {c_opt} - *{c_text_en}*\n"
            f"   (हिंदी: {c_text_hi})\n\n"
            f"💡 **Aasan Shabdon Mein:**\n"
            f"   Is prashn ka main concept '{section}' se related hai. "
            f"CPCT exam pattern ke anusaar, option {c_opt} hi sabse accurate aur scientifically correct definition hai. "
            f"Yeh component/concept system architecture mein directly yahi role perform karta hai.\n\n"
            f"❌ **Baki Options Kyon Galat Hain?**\n"
        )

        for opt_num in [1, 2, 3, 4]:
            if opt_num != c_opt:
                wrong_en = q.get(f"opt{opt_num}_en", "")
                exp += f"   - Option {opt_num} ({wrong_en[:50]}...): Yeh is question ke context mein match nahi karta ya dusre function se related hai.\n"

        exp += (
            f"\n⚡ **CPCT Exam Hack / Yaad Rakhne Ka Tarika:**\n"
            f"   Jab bhi '{section}' ke aise questions aayein, hamesha keywords par dhyan dein aur elimination method "
            f"(galat options ko pehle hatana) use karein! Is type ke questions exam mein baar-baar repeat hote hain."
        )
        return exp


class CPCTTrainer:
    """
    Adaptive training engine that selects questions based on weakness tracking,
    maintains scorecards, and provides mock exam simulations.
    """

    def __init__(self, db_path: Path = SQLITE_DB_PATH):
        self.db_path = Path(db_path)
        self.explainer = HinglishExplanationEngine()
        self._init_trainer_tables()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_trainer_tables(self):
        """Create tables for user attempt tracking and topic mastery."""
        with self._get_conn() as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS user_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                question_id INTEGER,
                selected_option INTEGER,
                correct_option INTEGER,
                is_correct INTEGER,
                time_spent_seconds REAL DEFAULT 0,
                attempted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS user_topic_mastery (
                section TEXT PRIMARY KEY,
                total_attempts INTEGER DEFAULT 0,
                correct_attempts INTEGER DEFAULT 0,
                accuracy REAL DEFAULT 0.0,
                mastery_level TEXT DEFAULT 'Beginner',
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            # Initialize default sections if not present
            for sec in SECTIONS:
                conn.execute("""
                INSERT OR IGNORE INTO user_topic_mastery (section, total_attempts, correct_attempts, accuracy, mastery_level)
                VALUES (?, 0, 0, 0.0, 'Beginner')
                """, (sec,))
            conn.commit()

    def reset_user_stats(self):
        """Reset all user attempts and topic mastery scores."""
        with self._get_conn() as conn:
            conn.execute("DELETE FROM user_attempts")
            conn.execute("UPDATE user_topic_mastery SET total_attempts = 0, correct_attempts = 0, accuracy = 0.0, mastery_level = 'Beginner'")
            conn.commit()

    def get_user_performance_summary(self) -> Dict[str, Any]:
        """Fetch overall stats, section-wise accuracy, and weakest areas."""
        with self._get_conn() as conn:
            # Overall stats
            overall = conn.execute("""
            SELECT COUNT(*) as total,
                   COALESCE(SUM(is_correct), 0) as correct,
                   COALESCE(AVG(time_spent_seconds), 0) as avg_time
            FROM user_attempts
            """).fetchone()

            total_q = overall["total"] if overall else 0
            correct_q = overall["correct"] if overall else 0
            overall_acc = round((correct_q / total_q) * 100, 1) if total_q > 0 else 0.0

            # Section breakdown
            sections_data = conn.execute("""
            SELECT section, total_attempts, correct_attempts, accuracy, mastery_level
            FROM user_topic_mastery
            ORDER BY accuracy ASC
            """).fetchall()

            section_list = [dict(s) for s in sections_data]
            # Identify weakest section (attempts > 0 with lowest accuracy, or with 0 attempts)
            weakest = section_list[0]["section"] if section_list else "Computer Proficiency"

            return {
                "total_attempted": total_q,
                "total_correct": correct_q,
                "overall_accuracy": overall_acc,
                "avg_time_per_q": round(overall["avg_time"], 1) if overall else 0,
                "weakest_section": weakest,
                "sections": section_list,
            }

    def record_attempt(
        self,
        question_id: int,
        selected_option: int,
        time_spent_seconds: float = 0.0,
        session_id: str = "default",
    ) -> Dict[str, Any]:
        """
        Record a user attempt, calculate correctness, update mastery,
        and generate Hinglish explanation.
        """
        with self._get_conn() as conn:
            q = conn.execute("SELECT * FROM questions WHERE id = ?", (question_id,)).fetchone()
            if not q:
                raise ValueError(f"Question with ID {question_id} not found.")

            q_dict = dict(q)
            is_correct = 1 if (selected_option == q_dict["correct_option"]) else 0

            # Record in user_attempts
            conn.execute("""
            INSERT INTO user_attempts (session_id, question_id, selected_option, correct_option, is_correct, time_spent_seconds)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (session_id, question_id, selected_option, q_dict["correct_option"], is_correct, time_spent_seconds))

            # Update section mastery
            sec = q_dict["section"]
            conn.execute("""
            UPDATE user_topic_mastery
            SET total_attempts = total_attempts + 1,
                correct_attempts = correct_attempts + ?,
                accuracy = ROUND(CAST(correct_attempts + ? AS REAL) / (total_attempts + 1) * 100, 1),
                mastery_level = CASE
                    WHEN (CAST(correct_attempts + ? AS REAL) / (total_attempts + 1)) >= 0.85 THEN 'Master'
                    WHEN (CAST(correct_attempts + ? AS REAL) / (total_attempts + 1)) >= 0.70 THEN 'Exam Ready'
                    WHEN (CAST(correct_attempts + ? AS REAL) / (total_attempts + 1)) >= 0.50 THEN 'Intermediate'
                    ELSE 'Needs Practice'
                END,
                last_updated = CURRENT_TIMESTAMP
            WHERE section = ?
            """, (is_correct, is_correct, is_correct, is_correct, is_correct, sec))
            conn.commit()

        # Generate Hinglish explanation
        explanation = self.explainer.generate_explanation(q_dict, user_choice=selected_option)

        return {
            "question_id": question_id,
            "is_correct": bool(is_correct),
            "correct_option": q_dict["correct_option"],
            "selected_option": selected_option,
            "explanation": explanation,
            "section": sec,
        }

    def get_next_question(
        self,
        mode: str = "adaptive",
        section: Optional[str] = None,
        exclude_ids: Optional[List[int]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch next adaptive question based on selected learning mode:
        - 'adaptive': Prioritizes user's weakest section and repeat questions
        - 'high_yield': Selects questions with highest repeat probabilities
        - 'section': Filters by a specific section
        - 'random': Uniform random selection
        """
        exclude_ids = exclude_ids or []
        exclude_clause = f"AND q.id NOT IN ({','.join(str(i) for i in exclude_ids)})" if exclude_ids else ""

        with self._get_conn() as conn:
            if mode == "high_yield":
                # High-yield repeating questions first
                query = f"""
                SELECT q.*, t.repeat_probability, t.repetition_count
                FROM questions q
                LEFT JOIN question_trends t ON cast(q.id as text) = t.question_id
                WHERE 1=1 {exclude_clause}
                ORDER BY t.repeat_probability DESC, RANDOM()
                LIMIT 1
                """
                row = conn.execute(query).fetchone()

            elif mode == "adaptive":
                # Check user weakest section
                perf = self.get_user_performance_summary()
                target_section = perf["weakest_section"]

                # 70% chance of weak section, 30% chance of high yield repeat
                if random.random() < 0.70 and target_section:
                    query = f"""
                    SELECT q.*, t.repeat_probability, t.repetition_count
                    FROM questions q
                    LEFT JOIN question_trends t ON cast(q.id as text) = t.question_id
                    WHERE q.section = ? {exclude_clause}
                    ORDER BY t.repeat_probability DESC, RANDOM()
                    LIMIT 1
                    """
                    row = conn.execute(query, (target_section,)).fetchone()
                else:
                    query = f"""
                    SELECT q.*, t.repeat_probability, t.repetition_count
                    FROM questions q
                    LEFT JOIN question_trends t ON cast(q.id as text) = t.question_id
                    WHERE 1=1 {exclude_clause}
                    ORDER BY t.repeat_probability DESC, RANDOM()
                    LIMIT 1
                    """
                    row = conn.execute(query).fetchone()

            elif mode == "section" and section:
                query = f"""
                SELECT q.*, t.repeat_probability, t.repetition_count
                FROM questions q
                LEFT JOIN question_trends t ON cast(q.id as text) = t.question_id
                WHERE q.section = ? {exclude_clause}
                ORDER BY RANDOM()
                LIMIT 1
                """
                row = conn.execute(query, (section,)).fetchone()

            else:
                # Default random
                query = f"""
                SELECT q.*, t.repeat_probability, t.repetition_count
                FROM questions q
                LEFT JOIN question_trends t ON cast(q.id as text) = t.question_id
                WHERE 1=1 {exclude_clause}
                ORDER BY RANDOM()
                LIMIT 1
                """
                row = conn.execute(query).fetchone()

            return dict(row) if row else None

    def generate_mock_exam(self) -> List[Dict[str, Any]]:
        """
        Generate a full 75-question Mock Exam strictly adhering to official CPCT proportions:
        - 52 Computer Proficiency
        - 5 Reading Comprehension
        - 6 Quantitative Aptitude
        - 6 Reasoning Aptitude
        - 6 General Awareness
        """
        mock_questions = []
        with self._get_conn() as conn:
            for section, count in SECTION_WEIGHTS.items():
                rows = conn.execute("""
                SELECT q.*, t.repeat_probability, t.repetition_count
                FROM questions q
                LEFT JOIN question_trends t ON cast(q.id as text) = t.question_id
                WHERE q.section = ?
                ORDER BY t.repeat_probability DESC, RANDOM()
                LIMIT ?
                """, (section, count)).fetchall()
                mock_questions.extend([dict(r) for r in rows])

        return mock_questions


def run_interactive_cli_quiz():
    """Terminal interactive personal trainer session using Rich UI."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt
    from rich.markdown import Markdown

    console = Console()
    trainer = CPCTTrainer()

    console.print(Panel.fit(
        "[bold cyan]🎯 MP CPCT AI Personal Trainer Coach[/bold cyan]\n"
        "[yellow]Adaptive bilingual quiz powered by Gemini 2.5 Flash & Semantic High-Yield Engine[/yellow]",
        border_style="green",
    ))

    # Show performance summary
    summary = trainer.get_user_performance_summary()
    perf_table = Table(title="📊 Your Section Mastery Dashboard", border_style="blue")
    perf_table.add_column("Section", style="bold")
    perf_table.add_column("Attempts", justify="right")
    perf_table.add_column("Accuracy", justify="right")
    perf_table.add_column("Status", style="green")

    for s in summary["sections"]:
        acc = f"{s['accuracy']}%" if s['total_attempts'] > 0 else "-"
        perf_table.add_row(s["section"], str(s["total_attempts"]), acc, s["mastery_level"])
    console.print(perf_table)

    console.print("\n[bold]Select Training Mode:[/bold]")
    console.print("1. [bold yellow]High-Yield Repeat Drill[/bold yellow] (Practice high-probability questions)")
    console.print("2. [bold cyan]Adaptive Weakness Trainer[/bold cyan] (Focus on your weakest areas)")
    console.print("3. [bold green]Full CPCT Mock Test[/bold green] (75 Questions timed simulation)")
    console.print("4. [bold magenta]Section-Wise Practice[/bold magenta] (Computer, Maths, Reasoning, GK, RC)")

    choice = Prompt.ask("Choose mode", choices=["1", "2", "3", "4"], default="1")
    mode_map = {"1": "high_yield", "2": "adaptive", "3": "mock", "4": "section"}
    selected_mode = mode_map[choice]

    target_sec = None
    if selected_mode == "section":
        console.print("\nAvailable Sections:")
        for idx, sec in enumerate(SECTIONS, 1):
            console.print(f"{idx}. {sec}")
        s_choice = Prompt.ask("Select Section", choices=[str(i) for i in range(1, len(SECTIONS) + 1)], default="1")
        target_sec = SECTIONS[int(s_choice) - 1]

    # Session loop
    score = 0
    total = 0
    exclude_ids = []

    console.print("\n[bold green]🚀 Quiz Starting! Type 1-4 to answer, or 'q' to quit.[/bold green]\n")

    while True:
        q = trainer.get_next_question(mode=selected_mode, section=target_sec, exclude_ids=exclude_ids)
        if not q:
            console.print("[yellow]No more questions available in this category![/yellow]")
            break

        exclude_ids.append(q["id"])
        total += 1

        prob = int(q.get("repeat_probability", 0) * 100)
        prob_tag = f"[bold red]🔥 Repeat Prob: {prob}%[/bold red]" if prob > 60 else ""

        # Display Question Card
        q_content = (
            f"[bold yellow]Q{total}. [{q['section']}][/bold yellow] {prob_tag}\n\n"
            f"[bold white]English:[/bold white] {q['question_en']}\n"
        )
        if q.get('question_hi'):
            q_content += f"[bold cyan]हिंदी:[/bold cyan] {q['question_hi']}\n"

        q_content += "\n[bold]Options:[/bold]\n"
        for i in [1, 2, 3, 4]:
            en_txt = q.get(f"opt{i}_en", "")
            hi_txt = q.get(f"opt{i}_hi", "")
            hi_part = f" | {hi_txt}" if hi_txt else ""
            q_content += f"  [bold]({i})[/bold] {en_txt}{hi_part}\n"

        console.print(Panel(q_content, border_style="cyan"))

        user_input = Prompt.ask("Your Answer", choices=["1", "2", "3", "4", "q"], default="1")
        if user_input.lower() == "q":
            break

        user_ans = int(user_input)
        result = trainer.record_attempt(q["id"], selected_option=user_ans)

        if result["is_correct"]:
            score += 1
            console.print("[bold green]✅ SAHI JAWAB! (Correct!)[/bold green]\n")
        else:
            console.print(f"[bold red]❌ GALAT JAWAB! Correct was Option {result['correct_option']}[/bold red]\n")

        # Display Hinglish Explanation
        console.print(Panel(Markdown(result["explanation"]), title="💡 AI Hinglish Coach Explanation", border_style="yellow"))
        console.print(f"[dim]Score: {score}/{total} ({score/total*100:.1f}%) | Press Enter for Next Question...[/dim]")
        input()

    console.print(f"\n[bold green]🏁 Session Complete! Final Score: {score}/{total}[/bold green]")


if __name__ == "__main__":
    run_interactive_cli_quiz()
