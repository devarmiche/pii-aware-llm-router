from src.router import route


def test_short_simple_query_routes_local() -> None:
    decision = route("Quel est le solde ?", "un court document", had_pii=False)
    assert decision.route == "local"


def test_pii_forces_local_even_if_otherwise_complex() -> None:
    decision = route(
        "Quelle est la jurisprudence applicable ?",
        "x" * 5000,
        had_pii=True,
        doc_count=3,
    )
    assert decision.route == "local"
    assert "PII" in decision.reason


def test_multi_doc_routes_api() -> None:
    decision = route("Compare ces documents", "doc court", had_pii=False, doc_count=2)
    assert decision.route == "api"
    assert "2 documents" in decision.reason


def test_single_doc_below_threshold_is_not_flagged_as_multi_doc() -> None:
    decision = route("Question simple", "doc court", had_pii=False, doc_count=1)
    assert decision.route == "local"


def test_long_document_routes_api() -> None:
    decision = route("Résume ce document", "a" * 5000, had_pii=False)
    assert decision.route == "api"
    assert "characters" in decision.reason


def test_legal_keyword_in_query_routes_api() -> None:
    decision = route(
        "Quelle est la jurisprudence sur ce point ?", "un court document", had_pii=False
    )
    assert decision.route == "api"
    assert "jurisprudence" in decision.reason


def test_legal_keyword_in_doc_context_routes_api() -> None:
    decision = route(
        "Explique ce document",
        "Ce contrat est soumis au code du travail.",
        had_pii=False,
    )
    assert decision.route == "api"


def test_keyword_matching_is_case_insensitive() -> None:
    decision = route("JURISPRUDENCE récente ?", "un court document", had_pii=False)
    assert decision.route == "api"


def test_default_local_when_no_signal_present() -> None:
    decision = route(
        "Quel est le montant total facturé ?",
        "Facture n°123, montant : 500 euros.",
        had_pii=False,
    )
    assert decision.route == "local"
