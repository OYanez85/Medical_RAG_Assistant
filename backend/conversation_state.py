from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Turn:
    user: str
    assistant: str
    matched_question: Optional[str] = None
    detected_task: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class ConversationState:
    history: List[Turn] = field(default_factory=list)
    active_topic: Optional[str] = None
    active_task: Optional[str] = None
    last_matched_question: Optional[str] = None
    last_confidence: Optional[float] = None

    def add_turn(self, user, assistant, matched_question=None, detected_task=None, confidence=None):
        self.history.append(
            Turn(
                user=user,
                assistant=assistant,
                matched_question=matched_question,
                detected_task=detected_task,
                confidence=confidence
            )
        )
        self.last_matched_question = matched_question
        self.active_task = detected_task
        self.last_confidence = confidence

    @classmethod
    def from_dict(cls, data: dict | None):
        if not data:
            return cls()

        raw_history = data.get("history", [])
        history = []
        for item in raw_history:
            if isinstance(item, Turn):
                history.append(item)
            else:
                history.append(Turn(**item))

        return cls(
            history=history,
            active_topic=data.get("active_topic"),
            active_task=data.get("active_task"),
            last_matched_question=data.get("last_matched_question"),
            last_confidence=data.get("last_confidence"),
        )
