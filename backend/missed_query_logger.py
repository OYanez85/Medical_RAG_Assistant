import os
import json
from datetime import datetime

LOG_PATH = os.path.join("data", "missed_queries.log")


def log_missed_query(
    question: str,
    confidence: float,
    matched_question: str | None,
    task: str | None,
):
    """
    Logs low-confidence or failed retrieval queries.
    This helps expand coverage weekly.
    """

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "question": question,
        "confidence": confidence,
        "matched_question": matched_question,
        "task": task,
    }

    os.makedirs("data", exist_ok=True)

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
