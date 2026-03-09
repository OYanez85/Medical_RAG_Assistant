from pathlib import Path
import json
import numpy as np
import re
from sentence_transformers import SentenceTransformer

from backend.schemas import AskResponse, TopMatch
from src.safety_rules import (
    detect_emergency,
    detect_high_risk,
    emergency_message,
    low_confidence_message,
)

BASE = Path(__file__).resolve().parent.parent
EMB_PATH = BASE / "data" / "qa_vector_embeddings_v2.npy"
META_PATH = BASE / "data" / "qa_vector_metadata_v2.json"
ALIASES_PATH = BASE / "src" / "aliases.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 8
MIN_CONFIDENCE = 0.55
SUGGESTION_COUNT = 3
TASK_BONUS = 0.12


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def apply_aliases(question: str, aliases: dict) -> str:
    q = normalize(question)
    for k, v in aliases.items():
        q = q.replace(k, v)
    return q


def detect_task(question: str) -> str:
    q = normalize(question)
    if q.startswith("what is the difference between"):
        return "comparison"
    if q.startswith("what are the symptoms of") or q.startswith("what are symptoms of"):
        return "symptoms"
    if q.startswith("what causes"):
        return "causes"
    if q.startswith("how is") and q.endswith("treated?"):
        return "treatment"
    if q.startswith("what is ") or q.startswith("explain "):
        return "definition"
    return "other"


def cosine_scores(query_emb, matrix):
    return matrix @ query_emb


def rerank_with_task(rows, scores, detected_task):
    adjusted = []
    for i, base_score in enumerate(scores):
        row_task = rows[i].get("task", "other")
        bonus = TASK_BONUS if row_task == detected_task else 0.0
        adjusted.append((i, float(base_score + bonus), float(base_score), row_task))
    adjusted.sort(key=lambda x: x[1], reverse=True)
    return adjusted


def format_suggestions(rows, results, count=3):
    seen = set()
    suggestions = []
    for idx, adjusted_score, raw_score, row_task in results:
        instruction = rows[idx]["instruction"]
        if instruction not in seen:
            seen.add(instruction)
            suggestions.append(instruction)
        if len(suggestions) >= count:
            break
    return suggestions


class MedicalRAGEngine:
    def __init__(self):
        print(f"Loading embedding model: {MODEL_NAME}")
        self.model = SentenceTransformer(MODEL_NAME)

        print(f"Loading metadata from: {META_PATH}")
        self.rows = json.loads(META_PATH.read_text(encoding="utf-8"))

        print(f"Loading aliases from: {ALIASES_PATH}")
        self.aliases = json.loads(ALIASES_PATH.read_text(encoding="utf-8"))

        if EMB_PATH.exists():
            print(f"Loading cached embeddings from: {EMB_PATH}")
            self.embeddings = np.load(EMB_PATH)
        else:
            print("Cached embeddings not found. Building embeddings at startup...")
            texts = [
                f"Instruction: {row['instruction']}\nAnswer: {row['response']}"
                for row in self.rows
            ]
            self.embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True
            )
            print(f"Saving rebuilt embeddings to: {EMB_PATH}")
            np.save(EMB_PATH, self.embeddings)

        print(f"Loaded {len(self.rows)} QA entries.")
        print(f"Embeddings shape: {self.embeddings.shape}")

    def ask(self, question: str) -> AskResponse:
        emergency_hits = detect_emergency(question)
        high_risk_hits = detect_high_risk(question)

        detected_task = detect_task(question)
        normalized_question = apply_aliases(question, self.aliases)

        if high_risk_hits:
            return AskResponse(
                question=question,
                normalized_question=normalized_question,
                task=detected_task,
                safety_flag=True,
                safety_type="high_risk",
                confidence=1.0,
                matched_question=None,
                answer="This question may involve serious risk and should be handled by an emergency professional or crisis service.",
                top_matches=[],
                suggestions=[]
            )

        if emergency_hits:
            return AskResponse(
                question=question,
                normalized_question=normalized_question,
                task=detected_task,
                safety_flag=True,
                safety_type="emergency",
                confidence=1.0,
                matched_question=None,
                answer=emergency_message(),
                top_matches=[],
                suggestions=[]
            )

        query_emb = self.model.encode(
            [normalized_question],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]

        raw_scores = cosine_scores(query_emb, self.embeddings)
        reranked = rerank_with_task(self.rows, raw_scores, detected_task)
        results = reranked[:TOP_K]

        best_idx, best_adjusted_score, best_raw_score, best_task = results[0]
        best = self.rows[best_idx]

        top_matches = [
            TopMatch(
                instruction=self.rows[idx]["instruction"],
                task=row_task,
                adjusted_score=adjusted_score,
                raw_score=raw_score,
            )
            for idx, adjusted_score, raw_score, row_task in results
        ]

        if best_adjusted_score < MIN_CONFIDENCE:
            return AskResponse(
                question=question,
                normalized_question=normalized_question,
                task=detected_task,
                safety_flag=False,
                safety_type="low_confidence",
                confidence=best_adjusted_score,
                matched_question=None,
                answer=low_confidence_message(),
                top_matches=top_matches,
                suggestions=format_suggestions(self.rows, results, count=SUGGESTION_COUNT),
            )

        return AskResponse(
            question=question,
            normalized_question=normalized_question,
            task=detected_task,
            safety_flag=False,
            safety_type=None,
            confidence=best_adjusted_score,
            matched_question=best["instruction"],
            answer=best["response"],
            top_matches=top_matches,
            suggestions=[]
        )
