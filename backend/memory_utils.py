import re
from backend.conversation_state import ConversationState


def is_followup(question: str) -> bool:
    q = question.lower().strip().rstrip("?")

    followup_exact = {
        "what are the symptoms",
        "what are symptoms",
        "how is it treated",
        "how is this treated",
        "how is it managed",
        "what causes it",
        "why does it happen",
        "and complications",
        "what are the complications",
        "complications",
    }

    if q in followup_exact:
        return True

    followup_prefixes = [
        "and ",
        "what about",
        "how about",
    ]

    return any(q.startswith(x) for x in followup_prefixes)


def extract_topic_from_matched_question(matched_question: str | None) -> str | None:
    if not matched_question:
        return None

    q = matched_question.lower().strip().rstrip("?")

    patterns = [
        r"^what are the complications of (.+)$",
        r"^what is the difference between (.+) and (.+)$",
        r"^what are the symptoms of (.+)$",
        r"^what are symptoms of (.+)$",
        r"^what causes (.+)$",
        r"^how is (.+) treated$",
        r"^what is (.+)$",
        r"^explain (.+) in simple terms$",
        r"^explain (.+)$",
    ]

    for p in patterns:
        m = re.match(p, q)
        if m:
            if len(m.groups()) == 2:
                return f"{m.group(1).strip()} and {m.group(2).strip()}"
            return m.group(1).strip()

    return None


def rewrite_followup(question: str, topic: str | None) -> str:
    if not topic:
        return question

    q = question.lower().strip().rstrip("?")

    if q in {"what are the symptoms", "what are symptoms"}:
        return f"What are the symptoms of {topic}?"

    if q in {"how is it treated", "how is this treated", "how is it managed"}:
        return f"How is {topic} treated?"

    if q in {"what causes it", "why does it happen"}:
        return f"What causes {topic}?"

    if q in {"and complications", "what are the complications", "complications"}:
        return f"What are the complications of {topic}?"

    if q.startswith("and "):
        tail = q[4:].strip()
        return f"{tail} of {topic}?"

    return question


def build_augmented_query(question: str, state: ConversationState) -> str:
    if not is_followup(question):
        return question

    return rewrite_followup(question, state.active_topic)
