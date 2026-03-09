import csv
from pathlib import Path

from backend.medical_engine import MedicalRAGEngine

BASE = Path(__file__).resolve().parent.parent
BENCHMARK_PATH = BASE / "benchmark_questions.csv"


def safe_contains(text: str, fragment: str) -> bool:
    return fragment.lower() in text.lower()


def main():
    engine = MedicalRAGEngine()

    rows = []
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total = 0
    task_correct = 0
    match_correct = 0
    answer_correct = 0
    safety_correct = 0

    print("=" * 100)
    print("Medical RAG Evaluation")
    print("=" * 100)

    for row in rows:
        total += 1
        question = row["question"]
        expected_task = row["expected_task"]
        expected_match = row["expected_match"]
        expected_answer_contains = row["expected_answer_contains"]
        expect_safety = row["expect_safety"].strip().lower() == "true"

        result = engine.ask(question)

        got_task = result.task
        got_match = result.matched_question or ""
        got_answer = result.answer or ""
        got_safety = bool(result.safety_flag)

        task_ok = got_task == expected_task
        match_ok = (expected_match == "") or (got_match == expected_match)
        answer_ok = (expected_answer_contains == "") or safe_contains(got_answer, expected_answer_contains)
        safety_ok = got_safety == expect_safety

        task_correct += int(task_ok)
        match_correct += int(match_ok)
        answer_correct += int(answer_ok)
        safety_correct += int(safety_ok)

        status = "PASS" if all([task_ok, match_ok, answer_ok, safety_ok]) else "FAIL"

        print(f"\n[{status}] {question}")
        print(f"  expected task:   {expected_task}")
        print(f"  got task:        {got_task}")
        print(f"  expected match:  {expected_match}")
        print(f"  got match:       {got_match}")
        print(f"  expected answer contains: {expected_answer_contains}")
        print(f"  got answer:      {got_answer}")
        print(f"  expected safety: {expect_safety}")
        print(f"  got safety:      {got_safety}")
        print(f"  confidence:      {result.confidence:.3f}")

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Total questions:        {total}")
    print(f"Task accuracy:          {task_correct}/{total} = {task_correct/total:.2%}")
    print(f"Match accuracy:         {match_correct}/{total} = {match_correct/total:.2%}")
    print(f"Answer containment:     {answer_correct}/{total} = {answer_correct/total:.2%}")
    print(f"Safety accuracy:        {safety_correct}/{total} = {safety_correct/total:.2%}")


if __name__ == "__main__":
    main()
