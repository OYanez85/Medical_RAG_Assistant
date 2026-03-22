from backend.conversation_state import ConversationState


def compose_reasoned_answer(base_answer: str, task: str) -> str:
    if task == "symptoms":
        return f"The main symptoms are: {base_answer}"
    if task == "causes":
        return f"The main causes include: {base_answer}"
    if task == "treatment":
        return f"Treatment usually includes: {base_answer}"
    if task == "comparison":
        return f"The difference is: {base_answer}"
    if task == "complications":
        return f"Possible complications include: {base_answer}"
    return base_answer


def avoid_repetition(new_answer: str, state: ConversationState) -> str:
    if not state.history:
        return new_answer

    last = state.history[-1].assistant.lower().strip()
    if new_answer.lower().strip() == last:
        return new_answer + " This is related to the same topic as the previous question."

    return new_answer
