import csv
from pathlib import Path

import pytest

from eval import blind_grade, merge_grades


@pytest.fixture
def results_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    results_path = tmp_path / "question_results.csv"
    questions_path = tmp_path / "questions.csv"
    with results_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question_id", "route", "model", "answer"])
        writer.writeheader()
        writer.writerow(
            {"question_id": "q01", "route": "local", "model": "mistral:7b-instruct", "answer": "réponse locale"}
        )
        writer.writerow(
            {"question_id": "q01", "route": "api", "model": "mistralai/mistral-large-2512", "answer": "réponse api"}
        )
    with questions_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question_id", "complexity", "question"])
        writer.writeheader()
        writer.writerow(
            {"question_id": "q01", "complexity": "factual_lookup", "question": "Quelle est la question ?"}
        )

    monkeypatch.setattr(blind_grade, "RESULTS_PATH", results_path)
    monkeypatch.setattr(blind_grade, "QUESTIONS_PATH", questions_path)
    monkeypatch.setattr(blind_grade, "GRADING_SHEET_PATH", tmp_path / "grading_sheet.csv")
    monkeypatch.setattr(blind_grade, "GRADING_KEY_PATH", tmp_path / "grading_key.csv")
    return tmp_path


def test_grading_sheet_hides_route(results_csv: Path) -> None:
    blind_grade.main()

    with blind_grade.GRADING_SHEET_PATH.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert "route" not in reader.fieldnames
        assert "model" not in reader.fieldnames
        answers = {row["answer"] for row in reader}

    assert answers == {"réponse locale", "réponse api"}


def test_grading_key_maps_back_to_route(results_csv: Path) -> None:
    blind_grade.main()

    with blind_grade.GRADING_SHEET_PATH.open(encoding="utf-8") as f:
        sheet = {row["grade_id"]: row for row in csv.DictReader(f)}
    with blind_grade.GRADING_KEY_PATH.open(encoding="utf-8") as f:
        key = {row["grade_id"]: row for row in csv.DictReader(f)}

    for grade_id, sheet_row in sheet.items():
        key_row = key[grade_id]
        expected_answer = "réponse locale" if key_row["route"] == "local" else "réponse api"
        assert sheet_row["answer"] == expected_answer


def test_merge_grades_raises_on_missing_score(
    results_csv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blind_grade.main()
    monkeypatch.setattr(merge_grades, "GRADING_SHEET_PATH", blind_grade.GRADING_SHEET_PATH)
    monkeypatch.setattr(merge_grades, "GRADING_KEY_PATH", blind_grade.GRADING_KEY_PATH)
    monkeypatch.setattr(merge_grades, "QUESTIONS_PATH", results_csv / "questions.csv")
    monkeypatch.setattr(merge_grades, "GRADED_RESULTS_PATH", results_csv / "graded_results.csv")

    with pytest.raises(ValueError, match="have no score yet"):
        merge_grades.main()


def test_merge_grades_computes_average_by_route(
    results_csv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blind_grade.main()
    monkeypatch.setattr(merge_grades, "GRADING_SHEET_PATH", blind_grade.GRADING_SHEET_PATH)
    monkeypatch.setattr(merge_grades, "GRADING_KEY_PATH", blind_grade.GRADING_KEY_PATH)
    monkeypatch.setattr(merge_grades, "QUESTIONS_PATH", results_csv / "questions.csv")
    monkeypatch.setattr(merge_grades, "GRADED_RESULTS_PATH", results_csv / "graded_results.csv")

    with blind_grade.GRADING_KEY_PATH.open(encoding="utf-8") as f:
        key = {row["grade_id"]: row["route"] for row in csv.DictReader(f)}

    rows = list(csv.DictReader(blind_grade.GRADING_SHEET_PATH.open(encoding="utf-8")))
    for row in rows:
        row["score"] = "3" if key[row["grade_id"]] == "api" else "1"
    with blind_grade.GRADING_SHEET_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["grade_id", "question_id", "question", "answer", "score"])
        writer.writeheader()
        writer.writerows(rows)

    merge_grades.main()

    graded = list(csv.DictReader((results_csv / "graded_results.csv").open(encoding="utf-8")))
    scores_by_route = {row["route"]: int(row["score"]) for row in graded}
    assert scores_by_route["api"] == 3
    assert scores_by_route["local"] == 1
