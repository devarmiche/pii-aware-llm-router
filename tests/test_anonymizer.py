from src.recognizers_fr import FrNirRecognizer, FrSiretRecognizer, _fr_nir_ok, _fr_siret_ok

# --- pure checksum functions: fast, no model load ---


def test_fr_nir_valid_checksum() -> None:
    assert _fr_nir_ok("275050105377821")


def test_fr_nir_rejects_bad_checksum() -> None:
    assert not _fr_nir_ok("275050105377820")


def test_fr_nir_rejects_wrong_length() -> None:
    assert not _fr_nir_ok("12345")


def test_fr_siret_valid_checksum() -> None:
    assert _fr_siret_ok("525 534 194 00812")


def test_fr_siret_rejects_bad_checksum() -> None:
    assert not _fr_siret_ok("525 534 194 00813")


def test_fr_siret_rejects_wrong_length() -> None:
    assert not _fr_siret_ok("12345")


# --- individual recognizers, no NLP needed (pure regex + checksum) ---


def test_fr_nir_recognizer_finds_valid_nir() -> None:
    recognizer = FrNirRecognizer()
    results = recognizer.analyze(
        "NIR: 275050105377821", entities=["FR_NIR"], nlp_artifacts=None
    )
    assert len(results) == 1
    assert results[0].score == 1.0


def test_fr_nir_recognizer_drops_invalid_checksum() -> None:
    recognizer = FrNirRecognizer()
    results = recognizer.analyze(
        "NIR: 275050105377820", entities=["FR_NIR"], nlp_artifacts=None
    )
    assert results == []


def test_fr_siret_recognizer_finds_valid_siret() -> None:
    recognizer = FrSiretRecognizer()
    results = recognizer.analyze(
        "SIRET : 525 534 194 00812", entities=["FR_SIRET"], nlp_artifacts=None
    )
    assert len(results) == 1
    assert results[0].score == 1.0


# --- full pipeline: loads fr_core_news_md once per test session ---

from src.anonymizer import anonymize  # noqa: E402


def test_anonymize_masks_and_maps_person() -> None:
    masked, mapping = anonymize("Jean Dupont habite à Lyon.")
    assert "Jean Dupont" not in masked
    assert mapping["[PERSON_1]"] == "Jean Dupont"


def test_anonymize_dedupes_repeated_entity() -> None:
    masked, mapping = anonymize("Jean Dupont a signé. Jean Dupont a confirmé.")
    assert masked.count("[PERSON_1]") == 2
    assert "[PERSON_2]" not in masked


def test_anonymize_catches_salary_without_thousand_separator() -> None:
    # "Rémunération" itself is sometimes misread as a LOCATION by the French
    # NER model (see eval/eval_anonymizer.py for measured limitations) —
    # this test only asserts the salary amount itself is caught.
    masked, mapping = anonymize("Rémunération mensuelle brute : 6264€.")
    assert "6264€" not in masked
    assert "6264€" in mapping.values()


def test_anonymize_ignores_unboosted_amount() -> None:
    # No salary-related context word nearby: score stays at the unboosted
    # base (0.2), below _MIN_SCORE["SALARY"], so it should not be masked.
    masked, _ = anonymize("Montant du marché : 145 000 €.")
    assert "145 000 €" in masked


def test_anonymize_masks_iban_and_email() -> None:
    text = "IBAN : FR9626542351161559407816184, contact hortense31@example.org"
    masked, mapping = anonymize(text)
    assert "FR9626542351161559407816184" not in masked
    assert "hortense31@example.org" not in masked
