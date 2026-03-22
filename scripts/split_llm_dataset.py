import json
import random
from pathlib import Path

random.seed(42)

src = Path("data/llm_training_dataset_expanded.jsonl")
train_dst = Path("data/llm_train.jsonl")
val_dst = Path("data/llm_val.jsonl")

rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
random.shuffle(rows)

split_idx = int(len(rows) * 0.9)
train_rows = rows[:split_idx]
val_rows = rows[split_idx:]

with train_dst.open("w", encoding="utf-8") as f:
    for row in train_rows:
        json.dump(row, f, ensure_ascii=False)
        f.write("\n")

with val_dst.open("w", encoding="utf-8") as f:
    for row in val_rows:
        json.dump(row, f, ensure_ascii=False)
        f.write("\n")

print("Train:", len(train_rows))
print("Val:", len(val_rows))
