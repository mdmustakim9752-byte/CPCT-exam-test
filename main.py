"""
Root Runner for MP CPCT AI Agent
Delegates execution to cpct_ai_agent/main.py
"""

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent
agent_dir = root_dir / "cpct_ai_agent"
for p in [str(root_dir), str(agent_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from cpct_ai_agent.main import run_cli, app

if __name__ == "__main__":
    run_cli()

