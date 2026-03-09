import os
import json
import re
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from src.safety_rules import check_safety
from backend.missed_query_logger import log_missed_query


@dataclass
class TopMatch:
    instruction: str
    task: str
    raw_score: float
    adjusted_score: float


@dataclass
class AskResponse:
    question: str
    normalized_question: str
    task: str
    safety_flag: bool
    safety_type: Optional[str]
    confidence: float
    matched_question: Optional[str]
    answer: str
    top_matches: List[TopMatch]
    suggestions: List[str]


def detect_task(question: str) -> str:
    q = question.lower().strip()

    if "difference between" in q:
        return "comparison"

    if (
        "symptom" in q
        or "signs of" in q
        or q.startswith("what are the symptoms of")
        or q.startswith("what are symptoms of")
    ):
        return "symptoms"

    if "cause" in q or "why does" in q or q.startswith("what causes"):
        return "causes"

    if (
        "treat" in q
        or "treatment of" in q
        or q.startswith("how is")
        or q.startswith("how can")
    ):
        return "treatment"

    if q.startswith("what is") or q.startswith("explain"):
        return "definition"

    return "definition"


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> List[str]:
    return normalize_text(text).split()


def cosine_similarity(a, b) -> float:
    return float(np.dot(a, b))


def minmax_scale(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    min_val = float(scores.min())
    max_val = float(scores.max())
    if max_val - min_val < 1e-8:
        return np.zeros_like(scores)
    return (scores - min_val) / (max_val - min_val)


def extract_comparison_terms(normalized_question: str) -> List[str]:
    q = normalized_question.lower()
    q = q.replace("what is the difference between", "")
    q = q.replace("difference between", "")
    parts = q.split(" and ")
    return [p.strip() for p in parts if p.strip()][:2]


def extract_primary_concept(normalized_question: str, detected_task: str) -> str:
    q = normalized_question.lower().strip()

    patterns = {
        "definition": [
            r"^what is (.+)$",
            r"^explain (.+)$",
        ],
        "symptoms": [
            r"^what are the symptoms of (.+)$",
            r"^what are symptoms of (.+)$",
            r"^what are the signs of (.+)$",
            r"^signs of (.+)$",
        ],
        "causes": [
            r"^what causes (.+)$",
            r"^why does (.+) happen$",
            r"^why does (.+) occur$",
        ],
        "treatment": [
            r"^how is (.+) treated$",
            r"^how can (.+) be treated$",
            r"^treatment of (.+)$",
        ],
    }

    for pattern in patterns.get(detected_task, []):
        m = re.match(pattern, q)
        if m:
            return m.group(1).strip()

    return ""


def format_suggestions(rows, results, detected_task, count=3):
    suggestions = []
    seen = set()

    for r in results[1:]:
        row = rows[r["index"]]
        if row["task"] == detected_task and row["instruction"] not in seen:
            suggestions.append(row["instruction"])
            seen.add(row["instruction"])
        if len(suggestions) >= count:
            return suggestions

    for r in results[1:]:
        row = rows[r["index"]]
        if row["instruction"] not in seen:
            suggestions.append(row["instruction"])
            seen.add(row["instruction"])
        if len(suggestions) >= count:
            return suggestions

    return suggestions


def low_confidence_message():
    return (
        "I do not have a reliable answer in my curated medical dataset, "
        "so I should not guess. Please rephrase the question or ask about "
        "a more specific condition."
    )


class MedicalRAGEngine:
    def __init__(self):
        metadata_path = "data/qa_vector_metadata_v2.json"
        if not os.path.exists(metadata_path):
            raise FileNotFoundError("qa_vector_metadata_v2.json not found")

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.rows = json.load(f)

        print(f"Loaded {len(self.rows)} QA entries.")

        alias_path = "src/aliases.json"
        if os.path.exists(alias_path):
            with open(alias_path, "r", encoding="utf-8") as f:
                self.alias_map = json.load(f)
        else:
            self.alias_map = {}

        self.embedding_model_name = os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedder = SentenceTransformer(self.embedding_model_name)

        vector_path = "data/qa_vector_embeddings_v2.npy"
        all_questions = [row["instruction"] for row in self.rows]

        if os.path.exists(vector_path):
            print(f"Loading cached embeddings from: {os.path.abspath(vector_path)}")
            self.embeddings = np.load(vector_path)
        else:
            print("Cached embeddings not found. Building embeddings at startup...")
            self.embeddings = self.embedder.encode(
                all_questions,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            np.save(vector_path, self.embeddings)

        if len(self.rows) != self.embeddings.shape[0]:
            raise ValueError(
                f"Metadata / embedding mismatch: rows={len(self.rows)} "
                f"vs embeddings={self.embeddings.shape[0]}"
            )

        self.corpus_tokens = [tokenize(row["instruction"]) for row in self.rows]
        self.bm25 = BM25Okapi(self.corpus_tokens)

        # Precision-tuned weights
        self.semantic_weight = float(os.getenv("SEMANTIC_WEIGHT", "0.50"))
        self.lexical_weight = float(os.getenv("LEXICAL_WEIGHT", "0.45"))
        self.task_bonus = float(os.getenv("TASK_BONUS", "0.05"))
        self.low_confidence_threshold = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.45"))

        print(
            f"Hybrid retrieval weights -> "
            f"semantic={self.semantic_weight}, lexical={self.lexical_weight}, task_bonus={self.task_bonus}"
        )
        print(f"Embeddings shape: {self.embeddings.shape}")

    def apply_aliases(self, text: str) -> str:
        q = text.lower()
        for alias, canonical in self.alias_map.items():
            if alias in q:
                q = q.replace(alias, canonical)
        return q

    def normalize_with_aliases(self, text: str) -> str:
        return normalize_text(self.apply_aliases(text))

    def ask(self, question: str) -> AskResponse:
        safety_flag, safety_type = check_safety(question)

        if safety_flag:
            return AskResponse(
                question=question,
                normalized_question=question,
                task="safety",
                safety_flag=True,
                safety_type=safety_type,
                confidence=1.0,
                matched_question=None,
                answer=(
                    "This question includes symptoms or signs that may "
                    "require urgent medical attention. Please contact "
                    "emergency services or seek immediate medical care right away."
                ),
                top_matches=[],
                suggestions=[]
            )

        raw_normalized_question = normalize_text(question)
        normalized_question = self.normalize_with_aliases(question)
        detected_task = detect_task(normalized_question)
        primary_concept = extract_primary_concept(normalized_question, detected_task)

        query_embedding = self.embedder.encode(
            normalized_question,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        semantic_scores = np.array(
            [cosine_similarity(query_embedding, emb) for emb in self.embeddings],
            dtype=np.float32
        )
        semantic_scaled = minmax_scale(semantic_scores)

        query_tokens = tokenize(normalized_question)
        lexical_scores = np.array(self.bm25.get_scores(query_tokens), dtype=np.float32)
        lexical_scaled = minmax_scale(lexical_scores)

        raw_query_tokens = set(tokenize(raw_normalized_question))
        norm_query_tokens = set(query_tokens)
        comparison_terms = extract_comparison_terms(normalized_question) if detected_task == "comparison" else []

        results = []

        for i, row in enumerate(self.rows):
            instruction = row["instruction"]
            instruction_norm = normalize_text(instruction)
            instruction_alias_norm = self.normalize_with_aliases(instruction)

            row_token_set = set(tokenize(instruction_alias_norm))
            semantic_score = float(semantic_scaled[i])
            lexical_score = float(lexical_scaled[i])

            base_score = (
                self.semantic_weight * semantic_score
                + self.lexical_weight * lexical_score
            )

            adjusted_score = base_score

            # Task boost
            if row["task"] == detected_task:
                adjusted_score += self.task_bonus

            # Exact raw question match
            if instruction_norm == raw_normalized_question:
                adjusted_score += 0.30

            # Exact normalized match after aliases
            if instruction_alias_norm == normalized_question:
                adjusted_score += 0.30

            # Token overlap boost / penalty
            overlap = len(norm_query_tokens & row_token_set)
            if overlap >= 3:
                adjusted_score += 0.05
            elif overlap < 2:
                adjusted_score -= 0.10

            # Strong concept matching for non-comparison tasks
            if primary_concept:
                if primary_concept in instruction_alias_norm:
                    adjusted_score += 0.20
                elif row["task"] == detected_task:
                    adjusted_score -= 0.15

            # Better comparison handling: must include both concepts
            if detected_task == "comparison" and len(comparison_terms) == 2:
                if all(term in instruction_alias_norm for term in comparison_terms):
                    adjusted_score += 0.25
                elif row["task"] == "comparison":
                    adjusted_score -= 0.10

            # Exact acronym rescue for abbreviation-like queries such as ORIF
            raw_upper = question.strip().upper()
            if raw_upper in instruction.upper():
                adjusted_score += 0.20

            results.append({
                "index": i,
                "raw_score": float(base_score),
                "adjusted_score": float(adjusted_score),
            })

        results = sorted(results, key=lambda x: x["adjusted_score"], reverse=True)

        best = results[0]
        best_row = self.rows[best["index"]]
        best_confidence = float(best["adjusted_score"])

        top_matches = []
        for r in results[:5]:
            row = self.rows[r["index"]]
            top_matches.append(
                TopMatch(
                    instruction=row["instruction"],
                    task=row["task"],
                    raw_score=float(r["raw_score"]),
                    adjusted_score=float(r["adjusted_score"]),
                )
            )

        suggestions = format_suggestions(self.rows, results, detected_task, count=3)

        if best_confidence < self.low_confidence_threshold:
            log_missed_query(
                question=question,
                confidence=best_confidence,
                matched_question=None,
                task=detected_task,
            )
            return AskResponse(
                question=question,
                normalized_question=normalized_question,
                task=detected_task,
                safety_flag=False,
                safety_type=None,
                confidence=best_confidence,
                matched_question=None,
                answer=low_confidence_message(),
                top_matches=top_matches,
                suggestions=suggestions
            )

        return AskResponse(
            question=question,
            normalized_question=normalized_question,
            task=detected_task,
            safety_flag=False,
            safety_type=None,
            confidence=best_confidence,
            matched_question=best_row["instruction"],
            answer=best_row["response"],
            top_matches=top_matches,
            suggestions=suggestions
        )
