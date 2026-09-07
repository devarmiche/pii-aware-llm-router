from dataclasses import dataclass

from src.anonymizer import anonymize
from src.backends import OLLAMA_MODEL, call_api, call_local
from src.router import RouteDecision, route
from src.tracker import Timer, compute_api_cost, log_call

# Placeholder until the OpenRouter model choice is settled — see claude.md.
API_MODEL = "mistral-large-latest"


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


def answer_question(doc_text: str, query: str) -> PipelineResult:
    masked_doc, mapping = anonymize(doc_text)
    decision = route(query, masked_doc, had_pii=bool(mapping))
    prompt = build_prompt(query, masked_doc)

    with Timer() as timer:
        if decision.route == "local":
            model = OLLAMA_MODEL
            response = call_local(prompt)
            cost_eur = 0.0
        else:
            model = API_MODEL
            response = call_api(prompt, model)  # raises until OpenRouter is wired
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
