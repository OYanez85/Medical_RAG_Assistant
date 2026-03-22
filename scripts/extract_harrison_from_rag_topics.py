import json
import re
from pathlib import Path

HARRISON_FILE = "data/Harrison.txt"
TOPICS_FILE = "data/rag_diagnosis_topics_curated.json"
OUTPUT_FILE = "data/harrison_rag_topic_passages.jsonl"

WINDOW_CHARS = 3600
MAX_PASSAGES_PER_TOPIC = 3
MIN_PASSAGE_CHARS = 900

CLINICAL_HINTS = [
    "symptom", "symptoms", "sign", "signs", "diagnosis", "diagnostic",
    "treatment", "therapy", "management", "clinical", "presentation",
    "complication", "complications", "cause", "causes", "patients",
    "disease", "disorder", "risk", "prognosis", "manifestation",
    "pathophysiology", "epidemiology"
]

BAD_PATTERNS = [
    r"\bEditor\b",
    r"\bEditions\b",
    r"\bProfessor of Medicine\b",
    r"\baccessmedicine\b",
    r"\btable of contents\b",
    r"\bcontributors\b",
    r"\bpreface\b",
    r"\bvideo library\b",
    r"\bonline edition\b",
    r"\bfurther reading\b",
    r"\bwww\.",
    r"\bN Engl J Med\b",
    r"\bJAMA\b",
    r"\bLancet\b",
    r"\bCirculation\b",
    r"\bEur Heart J\b",
]

def normalize_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text

def clean_passage(text: str) -> str:
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def looks_like_noise(text: str) -> bool:
    for p in BAD_PATTERNS:
        if re.search(p, text, flags=re.IGNORECASE):
            return True

    sentence_marks = text.count(".") + text.count(";") + text.count(":")
    if sentence_marks < 5:
        return True

    return False

def topic_mentions(text: str, topic: str) -> int:
    return len(re.findall(rf"\b{re.escape(topic.lower())}\b", text.lower()))

def passage_score(text: str, topic: str) -> int:
    t = text.lower()
    score = 0

    mentions = topic_mentions(text, topic)
    score += mentions * 6

    for hint in CLINICAL_HINTS:
        if hint in t:
            score += 1

    # Strong bonus if topic appears near the start of the passage
    early = t[:500]
    if re.search(rf"\b{re.escape(topic.lower())}\b", early):
        score += 5

    # Penalize references / noisy segments
    if looks_like_noise(text):
        score -= 30

    if len(text) < MIN_PASSAGE_CHARS:
        score -= 10

    return score

def extract_topic_passages(full_text: str, topic: str):
    candidates = []
    seen = set()

    pattern = re.compile(rf"\b{re.escape(topic)}\b", flags=re.IGNORECASE)

    for match in pattern.finditer(full_text):
        start = max(0, match.start() - WINDOW_CHARS // 2)
        end = min(len(full_text), match.end() + WINDOW_CHARS // 2)

        passage = clean_passage(full_text[start:end])

        key = passage[:600]
        if key in seen:
            continue
        seen.add(key)

        mentions = topic_mentions(passage, topic)
        if mentions < 2:
            continue

        score = passage_score(passage, topic)
        if score >= 15:
            candidates.append((score, passage))

    candidates.sort(key=lambda x: x[0], reverse=True)

    passages = []
    for score, passage in candidates[:MAX_PASSAGES_PER_TOPIC]:
        passages.append(passage)

    return passages

def main():
    text = Path(HARRISON_FILE).read_text(encoding="utf-8", errors="ignore")
    text = normalize_text(text)

    topics = json.loads(Path(TOPICS_FILE).read_text(encoding="utf-8"))

    written = 0
    topic_hits = {}

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for topic in topics:
            passages = extract_topic_passages(text, topic)
            topic_hits[topic] = len(passages)

            for i, passage in enumerate(passages, start=1):
                row = {
                    "id": f"{topic.replace(' ', '_')}_{i:03d}",
                    "source": "Harrison.txt",
                    "topic": topic,
                    "text": passage,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1

    print(f"Topics scanned: {len(topics)}")
    print(f"Passages saved: {written}")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\nTop matches:")
    top = sorted(topic_hits.items(), key=lambda x: x[1], reverse=True)[:40]
    for topic, n in top:
        print(f"{topic}: {n}")

    print("\nZero-hit topics:")
    zero = [topic for topic, n in topic_hits.items() if n == 0]
    for topic in zero[:80]:
        print("-", topic)

if __name__ == "__main__":
    main()
