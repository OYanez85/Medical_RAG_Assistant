from pydantic import BaseModel, Field
from typing import List, Optional


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Medical question from the user")


class TopMatch(BaseModel):
    instruction: str
    task: str
    adjusted_score: float
    raw_score: float


class AskResponse(BaseModel):
    question: str
    normalized_question: str
    task: str
    safety_flag: bool
    safety_type: Optional[str] = None
    confidence: float
    matched_question: Optional[str] = None
    answer: str
    top_matches: List[TopMatch]
    suggestions: List[str] = []
