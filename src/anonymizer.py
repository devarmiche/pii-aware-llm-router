from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider

from src.recognizers_fr import FrNirRecognizer, FrSiretRecognizer, SalaryRecognizer

_NLP_CONFIG = {
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "fr", "model_name": "fr_core_news_md"}],
}

_nlp_engine = NlpEngineProvider(nlp_configuration=_NLP_CONFIG).create_engine()

_registry = RecognizerRegistry(supported_languages=["fr"])
_registry.load_predefined_recognizers(languages=["fr"], nlp_engine=_nlp_engine)
_registry.add_recognizer(FrNirRecognizer())
_registry.add_recognizer(FrSiretRecognizer())
_registry.add_recognizer(SalaryRecognizer())

analyzer = AnalyzerEngine(
    registry=_registry, nlp_engine=_nlp_engine, supported_languages=["fr"]
)


# Provisional, not yet validated against the gold set — see eval/eval_anonymizer.py.
# Needed because SALARY's base regex score (0.2) matches any "<amount>€", and only
# the context-word boost (rémunération, brut, ...) tells a salary apart from a
# contract value; without this floor, unboosted matches get masked too.
_MIN_SCORE: dict[str, float] = {
    "SALARY": 0.35,
}


def _resolve_overlaps(results: list[RecognizerResult]) -> list[RecognizerResult]:
    accepted: list[RecognizerResult] = []
    for result in sorted(results, key=lambda r: -r.score):
        if not any(result.start < a.end and a.start < result.end for a in accepted):
            accepted.append(result)
    return sorted(accepted, key=lambda r: r.start)


def anonymize(text: str) -> tuple[str, dict[str, str]]:
    results = analyzer.analyze(text=text, language="fr")
    results = [r for r in results if r.score >= _MIN_SCORE.get(r.entity_type, 0.0)]
    results = _resolve_overlaps(results)

    mapping: dict[str, str] = {}
    placeholder_for: dict[tuple[str, str], str] = {}
    counters: dict[str, int] = {}

    out: list[str] = []
    cursor = 0
    for result in results:
        original = text[result.start : result.end]
        key = (result.entity_type, original)
        if key not in placeholder_for:
            counters[result.entity_type] = counters.get(result.entity_type, 0) + 1
            placeholder = f"[{result.entity_type}_{counters[result.entity_type]}]"
            placeholder_for[key] = placeholder
            mapping[placeholder] = original

        out.append(text[cursor : result.start])
        out.append(placeholder_for[key])
        cursor = result.end

    out.append(text[cursor:])
    return "".join(out), mapping
