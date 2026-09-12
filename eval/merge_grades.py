"""Reattach routes to a filled-in blind grading sheet and summarize quality
by route and by complexity — the answer-quality half of the router
evaluation described in claude.md step 5 (cost alone can't score a router).
"""

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

GRADING_SHEET_PATH = Path("eval/grading_sheet.csv")
GRADING_KEY_PATH = Path("eval/grading_key.csv")
QUESTIONS_PATH = Path("eval/questions.csv")
GRADED_RESULTS_PATH = Path("eval/graded_results.csv")


def main() -> None:
    with GRADING_SHEET_PATH.open(encoding="utf-8") as f:
        sheet = {row["grade_id"]: row for row in csv.DictReader(f)}

    with GRADING_KEY_PATH.open(encoding="utf-8") as f:
        key = {row["grade_id"]: row for row in csv.DictReader(f)}

    with QUESTIONS_PATH.open(encoding="utf-8") as f:
        complexity_by_id = {row["question_id"]: row["complexity"] for row in csv.DictReader(f)}

    ungraded = [gid for gid, row in sheet.items() if not row["score"].strip()]
    if ungraded:
        raise ValueError(
            f"{len(ungraded)} rows in {GRADING_SHEET_PATH} have no score yet: {ungraded}"
        )

    merged = []
    for grade_id, sheet_row in sheet.items():
        key_row = key[grade_id]
        score = int(sheet_row["score"])
        if score not in (1, 2, 3):
            raise ValueError(f"grade_id {grade_id}: score must be 1, 2 or 3, got {score!r}")
        merged.append(
            {
                "question_id": sheet_row["question_id"],
                "complexity": complexity_by_id[sheet_row["question_id"]],
                "route": key_row["route"],
                "model": key_row["model"],
                "score": score,
            }
        )

    with GRADED_RESULTS_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question_id", "complexity", "route", "model", "score"])
        writer.writeheader()
        writer.writerows(merged)

    print(f"{len(merged)} graded answers written to {GRADED_RESULTS_PATH}\n")

    by_route: dict[str, list[int]] = defaultdict(list)
    by_complexity_route: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in merged:
        by_route[row["route"]].append(row["score"])
        by_complexity_route[(row["complexity"], row["route"])].append(row["score"])

    print("Average score by route:")
    for route_name, scores in sorted(by_route.items()):
        print(f"  {route_name:<8} n={len(scores):<3} avg={mean(scores):.2f}")

    print("\nAverage score by complexity x route:")
    for (complexity, route_name), scores in sorted(by_complexity_route.items()):
        print(f"  {complexity:<20} {route_name:<8} n={len(scores):<3} avg={mean(scores):.2f}")


if __name__ == "__main__":
    main()
