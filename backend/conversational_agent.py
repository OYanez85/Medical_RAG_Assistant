from backend.medical_engine import MedicalRAGEngine
from backend.conversation_state import ConversationState
from backend.memory_utils import build_augmented_query, extract_topic_from_matched_question
from backend.reasoning_layer import compose_reasoned_answer, avoid_repetition


class ConversationalMedicalAgent:
    def __init__(self):
        self.engine = MedicalRAGEngine()
        self.memory_update_threshold = 0.90

    def ask(self, question: str, state: ConversationState):
        augmented_query = build_augmented_query(question, state)

        result = self.engine.ask(augmented_query)

        # Only update active topic if retrieval is strong and not safety
        if (
            not result.safety_flag
            and result.matched_question
            and result.confidence >= self.memory_update_threshold
        ):
            topic = extract_topic_from_matched_question(result.matched_question)
            if topic:
                state.active_topic = topic

        final_answer = compose_reasoned_answer(result.answer, result.task)
        final_answer = avoid_repetition(final_answer, state)

        state.add_turn(
            user=question,
            assistant=final_answer,
            matched_question=result.matched_question,
            detected_task=result.task,
            confidence=result.confidence
        )

        return result, final_answer, state, augmented_query
