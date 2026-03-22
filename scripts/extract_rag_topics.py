import json
import re
from pathlib import Path

INPUT_FILE = "data/qa_vector_metadata_v2_clean.json"
OUTPUT_FILE = "data/rag_diagnosis_topics.json"

QUESTION_PATTERNS = [
    r"^What is (.+?)\?$",
    r"^What are symptoms of (.+?)\?$",
    r"^What are the symptoms of (.+?)\?$",
    r"^What causes (.+?)\?$",
    r"^How is (.+?) treated\?$",
    r"^What are the complications of (.+?)\?$",
    r"^Explain (.+?) in simple terms\.$",
]

ALIASES = {
    "heart attack": "myocardial infarction",
    "lung clot": "pulmonary embolism",
    "collapsed lung": "pneumothorax",
    "tb": "tuberculosis",
    "copd": "chronic obstructive pulmonary disease",
    "high blood pressure": "hypertension",
    "low blood pressure": "hypotension",
    "overactive thyroid": "hyperthyroidism",
    "underactive thyroid": "hypothyroidism",
    "stomach ulcer": "gastric ulcer",
    "hansen disease": "leprosy",
    "echinococcosis": "hydatid disease",
    "ischaemic stroke": "ischemic stroke",
    "haemorrhagic stroke": "hemorrhagic stroke",
    "seizure disorder": "epilepsy",
    "oedema": "edema",
    "diarrhoea": "diarrhea",
    "anaemia": "anemia",
}

EXCLUDE_EXACT = {
    "unknown disease not in my dataset",
    "blablablitis",
    "emergency presentation",
}

def normalize_topic(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .,:;!?\"'`()[]{}#")

    # drop leading articles
    text = re.sub(r"^(a|an|the)\s+", "", text)

    text = ALIASES.get(text, text)
    return text

def looks_like_diagnosis(topic: str) -> bool:
    if not topic or len(topic) < 3:
        return False

    if topic in EXCLUDE_EXACT:
        return False

    # exclude comparisons and generic prompts
    bad_substrings = [
        "difference between",
        "warning signs",
        "red flag",
        "symptoms of a condition requiring",
        "causes the need for",
        "used in treatment",
        "side effects",
    ]
    if any(x in topic for x in bad_substrings):
        return False

    # exclude procedures/tests/meds/devices
    bad_words = [
        "scan", "ultrasound", "mri", "ct", "x-ray", "biopsy", "laparoscopy",
        "laparotomy", "appendectomy", "colonoscopy", "endoscopy", "stent",
        "troponin", "crp", "d-dimer", "hemoglobin", "creatinine",
        "aspirin", "ibuprofen", "amoxicillin", "metformin", "insulin",
        "beta blocker", "ace inhibitor", "anticoagulant", "bronchodilator",
        "corticosteroid", "proton pump inhibitor", "cpr", "aed",
        "triage", "intubation", "defibrillation", "protocol", "therapy"
    ]
    if any(x in topic for x in bad_words):
        return False

    return True

def main():
    rows = json.loads(Path(INPUT_FILE).read_text(encoding="utf-8"))
    topics = set()

    for row in rows:
        q = (row.get("question") or "").strip()
        for pattern in QUESTION_PATTERNS:
            m = re.match(pattern, q, flags=re.IGNORECASE)
            if m:
                topic = normalize_topic(m.group(1))
                if looks_like_diagnosis(topic):
                    topics.add(topic)
                break

    final_topics = sorted(topics)

    Path(OUTPUT_FILE).write_text(
        json.dumps(final_topics, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"Diagnosis topics extracted: {len(final_topics)}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("\nSample:")
    for t in final_topics[:100]:
        print("-", t)

if __name__ == "__main__":
    main()
