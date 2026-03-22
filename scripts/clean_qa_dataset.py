import json
import re
from pathlib import Path

INPUT_FILE = "data/qa_vector_metadata_v2.json"
OUTPUT_FILE = "data/qa_vector_metadata_v2_clean.json"

def clean_answer(text: str) -> str:
    if not text:
        return text

    # Cut off leaked section headers / separators
    text = re.split(r"\s*#{3,}.*", text)[0]
    text = re.split(r"\s*SECTION\s+[A-Z].*", text, flags=re.IGNORECASE)[0]

    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text

def main():
    rows = json.loads(Path(INPUT_FILE).read_text(encoding="utf-8"))
    cleaned = []

    changed = 0
    for row in rows:
        row = dict(row)
        old = row.get("answer", "")
        new = clean_answer(old)
        if old != new:
            changed += 1
        row["answer"] = new
        cleaned.append(row)

    Path(OUTPUT_FILE).write_text(
        json.dumps(cleaned, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"Rows processed: {len(cleaned)}")
    print(f"Rows changed: {changed}")
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
