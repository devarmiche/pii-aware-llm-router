from dataclasses import dataclass
from typing import Literal

# Openly heuristic, not a classifier — see claude.md. Curated, not exhaustive;
# expected to be revised once eval/questions.csv exists and can validate it.
LEGAL_KEYWORDS = [
    "jurisprudence",
    "cassation",
    "contentieux",
    "règlement",
    "réglementaire",
    "directive",
    "décret",
    "code civil",
    "code du travail",
    "conseil d'état",
    "rgpd",
    "cnil",
    "litige",
]

# Provisional, not yet validated against eval/questions.csv (doesn't exist yet).
_LENGTH_THRESHOLD_CHARS = 4000


@dataclass
class RouteDecision:
    route: Literal["local", "api"]
    reason: str


def route(
    query: str, doc_context: str, had_pii: bool, doc_count: int = 1
) -> RouteDecision:
    """Decide local vs. api. had_pii and doc_count come from the caller:
    had_pii from whether anonymize() found anything in doc_context, doc_count
    from however many source documents doc_context was assembled from.
    """
    if had_pii:
        return RouteDecision(
            "local",
            "document contains detected PII: sovereignty gate overrides complexity",
        )

    if doc_count > 1:
        return RouteDecision("api", f"cross-referencing {doc_count} documents")

    if len(doc_context) > _LENGTH_THRESHOLD_CHARS:
        return RouteDecision(
            "api",
            f"document context exceeds {_LENGTH_THRESHOLD_CHARS} characters "
            f"({len(doc_context)})",
        )

    combined = f"{query} {doc_context}".lower()
    hit = next((kw for kw in LEGAL_KEYWORDS if kw in combined), None)
    if hit is not None:
        return RouteDecision("api", f"legal/technical keyword detected: {hit!r}")

    return RouteDecision("local", "short, non-technical, single document")
