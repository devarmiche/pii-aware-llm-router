"""Split eval/question_results.csv into a blind grading sheet and a key.

The grading sheet hides route/model/cost/latency so a human can score each
answer on a 3-point scale (1 = wrong/unusable, 2 = partially correct,
3 = correct and complete) without knowing which backend produced it — see
claude.md step 5. Grade the sheet, then run merge_grades.py to reattach the
route and compute quality-by-route.
"""

import csv
import random
from pathlib import Path

RESULTS_PATH = Path("eval/question_results.csv")
QUESTIONS_PATH = Path("eval/questions.csv")
GRADING_SHEET_PATH = Path("eval/grading_sheet.csv")
GRADING_KEY_PATH = Path("eval/grading_key.csv")

SHEET_FIELDNAMES = ["grade_id", "question_id", "question", "answer", "score"]
KEY_FIELDNAMES = ["grade_id", "question_id", "route", "model"]


def main() -> None:
    with RESULTS_PATH.open(encoding="utf-8") as f:
        results = list(csv.DictReader(f))

    with QUESTIONS_PATH.open(encoding="utf-8") as f:
        questions_by_id = {row["question_id"]: row["question"] for row in csv.DictReader(f)}

    order = list(range(len(results)))
    random.shuffle(order)

    sheet_rows = []
    key_rows = []
    for grade_id, i in enumerate(order, start=1):
        row = results[i]
        sheet_rows.append(
            {
                "grade_id": grade_id,
                "question_id": row["question_id"],
                "question": questions_by_id[row["question_id"]],
                "answer": row["answer"],
                "score": "",
            }
        )
        key_rows.append(
            {
                "grade_id": grade_id,
                "question_id": row["question_id"],
                "route": row["route"],
                "model": row["model"],
            }
        )

    with GRADING_SHEET_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SHEET_FIELDNAMES)
        writer.writeheader()
        writer.writerows(sheet_rows)

    with GRADING_KEY_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=KEY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(key_rows)

    print(f"{len(sheet_rows)} answers to grade written to {GRADING_SHEET_PATH}")
    print("Fill in the 'score' column (1-3), then run eval/merge_grades.py.")
    print(f"Do not open {GRADING_KEY_PATH} before grading — it reveals the route.")


if __name__ == "__main__":
    main()
