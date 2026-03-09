from pathlib import Path
import json
from datetime import datetime

BASE = Path(__file__).resolve().parent.parent
LOG_DIR = BASE / "logs"
LOG_DIR.mkdir(exist_ok=True)

JSONL_LOG = LOG_DIR / "queries.jsonl"
CSV_LOG = LOG_DIR / "queries.csv"


def log_query(payload: dict):
    payload = dict(payload)
    payload["timestamp"] = datetime.utcnow().isoformat()

    with open(JSONL_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    csv_header = [
        "timestamp",
        "question",
        "normalized_question",
        "task",
        "safety_flag",
        "safety_type",
        "confidence",
        "matched_question",
        "answer",
        "source",
    ]

    file_exists = CSV_LOG.exists()
    with open(CSV_LOG, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write(",".join(csv_header) + "\n")

        row = [
            str(payload.get("timestamp", "")),
            _safe_csv(payload.get("question", "")),
            _safe_csv(payload.get("normalized_question", "")),
            _safe_csv(payload.get("task", "")),
            str(payload.get("safety_flag", "")),
            _safe_csv(payload.get("safety_type", "")),
            str(payload.get("confidence", "")),
            _safe_csv(payload.get("matched_question", "")),
            _safe_csv(payload.get("answer", "")),
            _safe_csv(payload.get("source", "")),
        ]
        f.write(",".join(row) + "\n")


def _safe_csv(value):
    text = str(value).replace('"', '""').replace("\n", " ").replace("\r", " ")
    return f'"{text}"'
