from pathlib import Path
import json
from datetime import datetime

BASE = Path.home() / "projects/LLMs-from-scratch/my_projects/biomed_gpt_pubmed_qa"
LOG_PATH = BASE / "logs" / "queries.jsonl"

def log_query(payload: dict):
    payload = dict(payload)
    payload["timestamp"] = datetime.utcnow().isoformat()

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
