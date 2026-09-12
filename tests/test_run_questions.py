from pathlib import Path

import pytest

from eval import run_questions
from src import tracker
from src.backends import LLMResponse


@pytest.fixture(autouse=True)
def _isolated_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tracker, "LOG_PATH", tmp_path / "tracker_log.csv")


def test_load_doc_text_concatenates_multiple_docs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_questions, "CORPUS_DIR", tmp_path)
    (tmp_path / "a.txt").write_text("Texte A", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Texte B", encoding="utf-8")

    doc_text, doc_count = run_questions.load_doc_text("a;b")

    assert doc_count == 2
    assert "Texte A" in doc_text
    assert "Texte B" in doc_text


def test_load_doc_text_single_doc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_questions, "CORPUS_DIR", tmp_path)
    (tmp_path / "a.txt").write_text("Texte A", encoding="utf-8")

    doc_text, doc_count = run_questions.load_doc_text("a")

    assert doc_count == 1
    assert doc_text == "Texte A"


def test_run_one_route_local_deanonymizes_and_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        run_questions,
        "call_local",
        lambda prompt: LLMResponse(text="[PERSON_1] répond.", input_tokens=10, output_tokens=5),
    )

    result = run_questions.run_one_route(
        "local", "doc masqué", {"[PERSON_1]": "Jean Dupont"}, "question ?", "reason"
    )

    assert result["answer"] == "Jean Dupont répond."
    assert result["cost_eur"] == 0.0
    assert len(tracker._read_rows()) == 1


def test_run_one_route_api_computes_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.pipeline import API_MODEL

    monkeypatch.setitem(tracker.PRICE_EUR_PER_1K_TOKENS, API_MODEL, (0.001, 0.002))
    monkeypatch.setattr(
        run_questions,
        "call_api",
        lambda prompt, model: LLMResponse(text="réponse", input_tokens=1000, output_tokens=500),
    )

    result = run_questions.run_one_route("api", "doc masqué", {}, "question ?", "reason")

    assert result["model"] == API_MODEL
    assert result["cost_eur"] == pytest.approx(0.001 + 0.001)
