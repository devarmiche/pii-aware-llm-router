from dataclasses import dataclass
from typing import Literal

from src.anonymizer import anonymize
from src.backends import OLLAMA_MODEL, call_api, call_local
from src.router import RouteDecision, route
from src.tracker import Timer, compute_api_cost, log_call

# Served through OpenRouter — see claude.md and src/backends.py.
# Pinned to a dated snapshot (not the floating "mistral-large" alias) so the
# measured cost in the business case stays reproducible. "-2512" only exists
# as a :batch (async) variant on OpenRouter as of 2026-09-19; 2407 is the
# latest snapshot reachable through the synchronous chat/completions route.
API_MODEL = "mistralai/mistral-large-2407"


def build_prompt(query: str, doc_context: str) -> str:
    return (
        "Tu es un assistant qui répond uniquement à partir du document fourni. "
        "Si l'information n'est pas dans le document, dis-le clairement.\n\n"
        f"Document :\n{doc_context}\n\n"
        f"Question : {query}\n"
        "Réponse :"
    )


def _deanonymize(text: str, mapping: dict[str, str]) -> str:
    for placeholder, original in mapping.items():
        text = text.replace(placeholder, original)
    return text


@dataclass
class PipelineResult:
    masked_doc: str
    decision: RouteDecision
    model: str
    answer: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_eur: float


def answer_question(
    doc_text: str,
    query: str,
    doc_count: int = 1,
    force_route: Literal["local", "api"] | None = None,
) -> PipelineResult:
    """doc_count: how many source documents doc_text was assembled from —
    forwarded to the router's cross-referencing signal. See eval/run_questions.py
    for multi-document (cross-document) questions.

    force_route: bypass the router and use this route instead (the app's
    "100% Local" / "100% API" modes) — the auto decision is still computed
    and logged in the reason, for comparison."""
    masked_doc, mapping = anonymize(doc_text)
    auto_decision = route(query, masked_doc, had_pii=bool(mapping), doc_count=doc_count)
    if force_route is None:
        decision = auto_decision
    else:
        decision = RouteDecision(
            force_route,
            f"forced by user ({force_route}); auto decision was "
            f"{auto_decision.route} ({auto_decision.reason})",
        )
    prompt = build_prompt(query, masked_doc)

    with Timer() as timer:
        if decision.route == "local":
            model = OLLAMA_MODEL
            response = call_local(prompt)
            cost_eur = 0.0
        else:
            model = API_MODEL
            response = call_api(prompt, model)
            cost_eur = compute_api_cost(
                model, response.input_tokens, response.output_tokens
            )

    answer = _deanonymize(response.text, mapping)

    log_call(
        route=decision.route,
        model=model,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        latency_ms=timer.elapsed_ms,
        cost_eur=cost_eur,
        reason=decision.reason,
    )

    return PipelineResult(
        masked_doc=masked_doc,
        decision=decision,
        model=model,
        answer=answer,
        latency_ms=timer.elapsed_ms,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cost_eur=cost_eur,
    )
