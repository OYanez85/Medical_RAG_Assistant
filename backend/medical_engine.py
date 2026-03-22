import json
import os
import re
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class RetrievalResult:
    answer: str
    matched_question: str | None
    confidence: float
    task: str
    safety_flag: bool = False


def detect_task(question: str) -> str:
    q = question.lower().strip()

    if "difference between" in q:
        return "comparison"

    if "complication" in q:
        return "complications"

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


def infer_task_from_question_text(text: str | None) -> str | None:
    if not text:
        return None

    q = text.lower().strip()

    if "difference between" in q:
        return "comparison"
    if "complication" in q:
        return "complications"
    if "symptom" in q or "signs of" in q:
        return "symptoms"
    if "cause" in q or "why does" in q:
        return "causes"
    if "treat" in q or "treatment" in q:
        return "treatment"
    if q.startswith("what is") or q.startswith("explain"):
        return "definition"

    return None


def task_matches_request(requested_task: str, matched_question: str | None) -> bool:
    matched_task = infer_task_from_question_text(matched_question)
    if not matched_task:
        return False
    return matched_task == requested_task


def extract_topic_from_question(question: str) -> str | None:
    q = question.lower().strip().rstrip("?")

    patterns = [
        r"^what are the complications of (.+)$",
        r"^what are the symptoms of (.+)$",
        r"^what are symptoms of (.+)$",
        r"^what causes (.+)$",
        r"^how is (.+) treated$",
        r"^what is the difference between (.+) and (.+)$",
        r"^what is (.+)$",
        r"^explain (.+)$",
    ]

    for p in patterns:
        m = re.match(p, q)
        if m:
            if len(m.groups()) == 2:
                return f"{m.group(1).strip()} and {m.group(2).strip()}"
            return m.group(1).strip()

    return None


def build_task_mismatch_response(task: str, topic: str | None) -> str:
    subject = topic or "this condition"

    if task == "complications":
        return f"I could not find a reliable complications-specific answer for {subject} in the current knowledge base."
    if task == "symptoms":
        return f"I could not find a reliable symptoms-specific answer for {subject} in the current knowledge base."
    if task == "treatment":
        return f"I could not find a reliable treatment-specific answer for {subject} in the current knowledge base."
    if task == "causes":
        return f"I could not find a reliable causes-specific answer for {subject} in the current knowledge base."
    if task == "comparison":
        return f"I could not find a reliable comparison-specific answer for {subject} in the current knowledge base."

    return f"I could not find a reliable answer for {subject} in the current knowledge base."


def should_block_due_to_task_mismatch(
    requested_task: str,
    matched_question: str | None,
    confidence: float,
    complications_min_confidence: float = 0.72,
) -> bool:
    strict_tasks = {"complications", "symptoms", "treatment", "causes", "comparison"}

    if requested_task not in strict_tasks:
        return False

    if not task_matches_request(requested_task, matched_question):
        return True

    if requested_task == "complications" and confidence < complications_min_confidence:
        return True

    return False


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def simple_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9\-]+", text.lower())


def lexical_overlap_score(q1: str, q2: str) -> float:
    t1 = set(simple_tokens(q1))
    t2 = set(simple_tokens(q2))

    if not t1 or not t2:
        return 0.0

    return len(t1 & t2) / max(len(t1), 1)


def parse_qa_file(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    pattern = r"### Instruction:\s*(.*?)\s*### Response:\s*(.*?)(?=\n### Instruction:|\Z)"
    matches = re.findall(pattern, text, flags=re.DOTALL)

    qa_entries = []
    for instruction, response in matches:
        question = normalize_text(instruction)
        answer = normalize_text(response)

        if question and answer:
            qa_entries.append(
                {
                    "question": question,
                    "answer": answer,
                    "task": detect_task(question),
                }
            )

    return qa_entries


class MedicalRAGEngine:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_path = os.path.join(self.base_dir, "data", "phase1_manual_expansion_working.txt")
        self.embeddings_path = os.path.join(self.base_dir, "data", "qa_vector_embeddings_v2.npy")
        self.metadata_path = os.path.join(self.base_dir, "data", "qa_vector_metadata_v2.json")

        self.complications_min_confidence = 0.72

        print("Loading QA data from:", self.data_path)
        self.rows = parse_qa_file(self.data_path)
        print(f"Loaded {len(self.rows)} QA entries.")

        print("Loading embedding model: sentence-transformers/all-MiniLM-L6-v2")
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        self.embeddings = self._load_or_build_embeddings()
        print("Hybrid retrieval weights -> semantic=0.55, lexical=0.35, task_bonus=0.10")
        print("Embeddings shape:", self.embeddings.shape)

    def _load_or_build_embeddings(self) -> np.ndarray:
        current_questions = [row["question"] for row in self.rows]

        if os.path.exists(self.embeddings_path) and os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    cached_meta = json.load(f)

                cached_questions = [row["question"] for row in cached_meta]

                if cached_questions == current_questions:
                    print("Loading cached embeddings from:", self.embeddings_path)
                    return np.load(self.embeddings_path)
                else:
                    print("Cache mismatch detected. Rebuilding embeddings...")
            except Exception as e:
                print("Could not use cache. Rebuilding embeddings...", str(e))

        all_questions = [row["question"] for row in self.rows]
        print("Computing embeddings for QA dataset...")
        embeddings = self.model.encode(
            all_questions,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        np.save(self.embeddings_path, embeddings)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.rows, f, ensure_ascii=False, indent=2)

        print("Saved embeddings to:", self.embeddings_path)
        print("Saved metadata to:", self.metadata_path)
        return embeddings

    def _find_exact_match(self, question: str):
        q = normalize_text(question).lower()
        for row in self.rows:
            if row["question"].lower() == q:
                return row["question"], row["answer"], 1.50
        return None

    def retrieve(self, question: str):
        exact = self._find_exact_match(question)
        if exact is not None:
            return exact

        query_task = detect_task(question)
        query_topic = extract_topic_from_question(question)

        query_embedding = self.model.encode(
            [question],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        semantic_scores = self.embeddings @ query_embedding

        candidate_indices = []
        for i, row in enumerate(self.rows):
            row_question = row["question"]
            row_task = row["task"]

            if query_task == "definition":
                if row_task != "definition":
                    continue
            elif query_task in {"symptoms", "treatment", "causes", "comparison", "complications"}:
                if row_task != query_task:
                    continue

            if query_topic and query_topic.lower() not in row_question.lower():
                continue

            candidate_indices.append(i)

        if not candidate_indices:
            for i, row in enumerate(self.rows):
                if row["task"] == query_task:
                    candidate_indices.append(i)

        if not candidate_indices:
            candidate_indices = list(range(len(self.rows)))

        best_idx = None
        best_score = -1.0

        for i in candidate_indices:
            row = self.rows[i]
            semantic = float(semantic_scores[i])
            lexical = lexical_overlap_score(question, row["question"])
            task_bonus = 0.10 if row["task"] == query_task else 0.0

            topic_bonus = 0.0
            if query_topic and query_topic.lower() in row["question"].lower():
                topic_bonus = 0.10

            exact_prefix_bonus = 0.0
            q_lower = question.lower().strip().rstrip("?")
            r_lower = row["question"].lower().strip().rstrip("?")
            if q_lower == r_lower:
                exact_prefix_bonus = 0.25

            score = (
                0.55 * semantic
                + 0.35 * lexical
                + task_bonus
                + topic_bonus
                + exact_prefix_bonus
            )

            if score > best_score:
                best_score = score
                best_idx = i

        best = self.rows[best_idx]
        return best["question"], best["answer"], float(best_score)

    def ask(self, question: str):
        task = detect_task(question)

        matched_question, answer, confidence = self.retrieve(question)
        topic = extract_topic_from_question(question)

        if should_block_due_to_task_mismatch(
            requested_task=task,
            matched_question=matched_question,
            confidence=confidence,
            complications_min_confidence=self.complications_min_confidence,
        ):
            return RetrievalResult(
                answer=build_task_mismatch_response(task, topic),
                matched_question=matched_question,
                confidence=confidence,
                task=task,
                safety_flag=False,
            )

        return RetrievalResult(
            answer=answer,
            matched_question=matched_question,
            confidence=confidence,
            task=task,
            safety_flag=False,
        )
