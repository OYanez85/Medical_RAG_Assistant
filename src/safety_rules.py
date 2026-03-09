import re

EMERGENCY_PATTERNS = [
    r"\bchest pain\b",
    r"\bshortness of breath\b",
    r"\bdifficulty breathing\b",
    r"\bcannot breathe\b",
    r"\btrouble breathing\b",
    r"\bsevere bleeding\b",
    r"\buncontrolled bleeding\b",
    r"\bcoughing up blood\b",
    r"\bvomiting blood\b",
    r"\bblood in stool\b",
    r"\bloss of consciousness\b",
    r"\bunconscious\b",
    r"\bfainted\b",
    r"\bfainting\b",
    r"\bseizure\b",
    r"\bconvulsion\b",
    r"\bstroke\b",
    r"\bface drooping\b",
    r"\bslurred speech\b",
    r"\bone-sided weakness\b",
    r"\bsevere headache\b",
    r"\bworst headache\b",
    r"\bstiff neck\b",
    r"\banaphylaxis\b",
    r"\bswollen tongue\b",
    r"\bswollen throat\b",
    r"\bsepsis\b",
    r"\bconfusion\b",
    r"\bblue lips\b",
    r"\bcyanosis\b",
]

HIGH_RISK_PATTERNS = [
    r"\bsuicidal\b",
    r"\bkill myself\b",
    r"\bself harm\b",
    r"\boverdose\b",
]

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())

def detect_emergency(text: str):
    text = normalize(text)
    hits = []
    for pattern in EMERGENCY_PATTERNS:
        if re.search(pattern, text):
            hits.append(pattern)
    return hits

def detect_high_risk(text: str):
    text = normalize(text)
    hits = []
    for pattern in HIGH_RISK_PATTERNS:
        if re.search(pattern, text):
            hits.append(pattern)
    return hits

def emergency_message():
    return (
        "This question includes symptoms or signs that may require urgent medical attention. "
        "Please contact emergency services or seek immediate medical care right away."
    )

def low_confidence_message():
    return (
        "I do not have a reliable answer in my curated medical dataset, so I should not guess. "
        "Please rephrase the question or ask about a more specific condition."
    )
