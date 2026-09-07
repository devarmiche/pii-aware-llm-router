from pathlib import Path

import pytest

from src import pipeline, tracker
from src.backends import LLMResponse


@pytest.fixture(autouse=True)
def _isolated_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Same reasoning as test_tracker.py: never touch the real tracker_log.csv."""
    monkeypatch.setattr(tracker, "LOG_PATH", tmp_path / "tracker_log.csv")


def test_build_prompt_includes_query_and_doc() -> None:
    prompt = pipeline.build_prompt("Quel est le salaire ?", "Bonjour [PERSON_1]")
    assert "Quel est le salaire ?" in prompt
    assert "Bonjour [PERSON_1]" in prompt


def test_deanonymize_substitutes_placeholders() -> None:
    result = pipeline._deanonymize(
        "Le salaire de [PERSON_1] est [SALARY_1]",
        {"[PERSON_1]": "Jean Dupont", "[SALARY_1]": "3000€"},
    )
    assert result == "Le salaire de Jean Dupont est 3000€"


def test_deanonymize_leaves_unmapped_text_untouched() -> None:
    result = pipeline._deanonymize("Aucune entité ici.", {})
    assert result == "Aucune entité ici."


def test_answer_question_routes_local_and_deanonymizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pipeline,
        "call_local",
        lambda prompt: LLMResponse(
            text="L'email de contact est [EMAIL_ADDRESS_1].",
            input_tokens=42,
            output_tokens=8,
        ),
    )

    result = pipeline.answer_question(
        "Contactez-moi à jean.dupont@example.com pour plus d'infos.",
        "Quel est l'email de contact ?",
    )

    assert result.decision.route == "local"
    assert "jean.dupont@example.com" in result.answer
    assert "[EMAIL_ADDRESS_1]" not in result.answer
    assert "jean.dupont@example.com" not in result.masked_doc
    assert result.cost_eur == 0.0
    assert result.input_tokens == 42
    assert result.output_tokens == 8


def test_answer_question_logs_the_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pipeline,
        "call_local",
        lambda prompt: LLMResponse(text="réponse", input_tokens=10, output_tokens=5),
    )

    pipeline.answer_question("Document sans PII ni mots-clés.", "Résume ce document.")

    rows = tracker._read_rows()
    assert len(rows) == 1
    assert rows[0]["route"] == "local"
    assert rows[0]["model"] == pipeline.OLLAMA_MODEL
    assert rows[0]["cost_eur"] == "0.0"


def test_pii_gate_forces_local_even_with_legal_keyword(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pipeline,
        "call_local",
        lambda prompt: LLMResponse(text="réponse", input_tokens=10, output_tokens=5),
    )

    result = pipeline.answer_question(
        "Contactez-moi à jean.dupont@example.com.",
        "Quelle est la jurisprudence applicable ?",
    )

    assert result.decision.route == "local"
    assert "PII" in result.decision.reason


def test_answer_question_raises_when_routed_to_unimplemented_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(NotImplementedError):
        pipeline.answer_question(
            "Document sans PII.", "Quelle est la jurisprudence applicable ?"
        )

    # the crash happens before log_call, so nothing should have been written
    assert not tracker.LOG_PATH.exists()
