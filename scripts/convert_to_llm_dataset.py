import json
from pathlib import Path

src = Path("data/qa_vector_metadata_v2.json")
dst = Path("data/llm_training_dataset.jsonl")

rows = json.loads(src.read_text(encoding="utf-8"))

def build_prompt(row):
    return [
        {"role": "system", "content": "You are a careful medical assistant. Be accurate and structured."},
        {"role": "user", "content": row["question"]},
        {"role": "assistant", "content": row["answer"]}
    ]

with dst.open("w", encoding="utf-8") as f:
    for row in rows:
        json.dump({"messages": build_prompt(row)}, f)
        f.write("\n")

print("Saved:", dst)
