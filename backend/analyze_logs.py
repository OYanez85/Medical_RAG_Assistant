from pathlib import Path
import json
from collections import Counter

BASE = Path(__file__).resolve().parent.parent
LOG_PATH = BASE / "logs" / "queries.jsonl"


def main():
    if not LOG_PATH.exists():
        print("No logs found.")
        return

    rows = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    print(f"Total queries: {len(rows)}")

    task_counts = Counter(r.get("task", "unknown") for r in rows)
    safety_counts = Counter(r.get("safety_type") or "none" for r in rows)

    print("\nTasks:")
    for k, v in task_counts.most_common():
        print(f"- {k}: {v}")

    print("\nSafety types:")
    for k, v in safety_counts.most_common():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
