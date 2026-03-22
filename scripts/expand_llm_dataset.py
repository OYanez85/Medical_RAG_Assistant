import json
import re
from pathlib import Path

src = Path("data/qa_vector_metadata_v2.json")
dst = Path("data/llm_training_dataset_expanded.jsonl")

rows = json.loads(src.read_text(encoding="utf-8"))

SYSTEM_PROMPT = "You are a careful medical assistant. Be accurate and structured."


def extract_condition(question: str, task: str) -> str | None:
    q = question.strip().rstrip("?")

    patterns = {
        "definition": [
            r"^What is (.+)$",
            r"^Explain (.+)$",
        ],
        "symptoms": [
            r"^What are symptoms of (.+)$",
            r"^What are the symptoms of (.+)$",
        ],
        "causes": [
            r"^What causes (.+)$",
        ],
        "treatment": [
            r"^How is (.+) treated$",
        ],
        "complications": [
            r"^What are the complications of (.+)$",
        ],
    }

    for p in patterns.get(task, []):
        m = re.match(p, q, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()

    return None


def question_variants(condition: str, task: str) -> list[str]:
    if task == "definition":
        return [
            f"What is {condition}?",
            f"Explain {condition}.",
            f"Explain {condition} in simple terms.",
            f"What does {condition} mean?",
        ]

    if task == "symptoms":
        return [
            f"What are the symptoms of {condition}?",
            f"What are symptoms of {condition}?",
            f"What symptoms does {condition} cause?",
            f"How does {condition} present?",
            f"What are the signs of {condition}?",
        ]

    if task == "causes":
        return [
            f"What causes {condition}?",
            f"What are the causes of {condition}?",
            f"Why does {condition} happen?",
            f"What leads to {condition}?",
        ]

    if task == "treatment":
        return [
            f"How is {condition} treated?",
            f"What is the treatment for {condition}?",
            f"How do you treat {condition}?",
            f"How is {condition} managed?",
        ]

    if task == "complications":
        return [
            f"What are the complications of {condition}?",
            f"What complications can {condition} cause?",
            f"What can happen if {condition} gets worse?",
            f"What are the possible complications of {condition}?",
        ]

    return []


def build_messages(user_q: str, answer: str):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_q},
            {"role": "assistant", "content": answer},
        ]
    }


written = 0
seen = set()

with dst.open("w", encoding="utf-8") as f:
    for row in rows:
        question = row.get("question") or row.get("instruction")
        answer = row.get("answer") or row.get("response")
        task = row.get("task", "").strip().lower()

        if not question or not answer or not task:
            continue

        condition = extract_condition(question, task)
        if not condition:
            key = (question.strip(), answer.strip())
            if key not in seen:
                json.dump(build_messages(question, answer), f, ensure_ascii=False)
                f.write("\n")
                seen.add(key)
                written += 1
            continue

        for variant in question_variants(condition, task):
            key = (variant.strip(), answer.strip())
            if key in seen:
                continue
            json.dump(build_messages(variant, answer), f, ensure_ascii=False)
            f.write("\n")
            seen.add(key)
            written += 1

print("Saved:", dst)
print("Rows written:", written)
