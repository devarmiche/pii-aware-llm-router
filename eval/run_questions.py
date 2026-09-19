"""Run eval/questions.csv through both backends for the router/cost
measurement described in claude.md step 5.

For each question this scores the router's automatic decision against the
question's a priori complexity tag (logged, not enforced), then forces an
answer from *each* requested route so cost/latency/quality are comparable
question-for-question — not just whatever the router happened to pick.
Every call is still logged to data/tracker_log.csv via src.tracker.

Requires OPENROUTER_API_KEY in the environment for --routes api, and a
running Ollama with mistral:7b-instruct for --routes local.

Known limitation: several corpus documents (schrems_ii_*, vinci_semestriel_2025)
are tens of thousands of tokens; the local model's context window may not
fit them. Such failures are caught, logged to stderr, and skipped rather than
silently truncated.
"""

import argparse
import csv
from pathlib import Path

from src.anonymizer import anonymize
from src.backends import OLLAMA_MODEL, call_api, call_local
from src.pipeline import API_MODEL, _deanonymize, build_prompt
from src.router import route
from src.tracker import Timer, compute_api_cost, log_call

QUESTIONS_PATH = Path("eval/questions.csv")
CORPUS_DIR = Path("data/corpus")
RESULTS_PATH = Path("eval/question_results.csv")

FIELDNAMES = [
    "question_id",
    "complexity",
    "auto_route",
    "auto_reason",
    "route",
    "model",
    "answer",
    "latency_ms",
    "input_tokens",
    "output_tokens",
    "cost_eur",
]


def load_doc_text(doc_ids: str) -> tuple[str, int]:
    ids = doc_ids.split(";")
    texts = [
        (CORPUS_DIR / f"{doc_id}.txt").read_text(encoding="utf-8") for doc_id in ids
    ]
    return "\n\n---\n\n".join(texts), len(ids)


def run_one_route(
    route_name: str, masked_doc: str, mapping: dict[str, str], query: str, reason: str
) -> dict:
    prompt = build_prompt(query, masked_doc)
    with Timer() as timer:
        if route_name == "local":
            model = OLLAMA_MODEL
            response = call_local(prompt)
            cost_eur = 0.0
        else:
            model = API_MODEL
            response = call_api(prompt, model)
            cost_eur = compute_api_cost(
                model, response.input_tokens, response.output_tokens
            )

    log_call(
        route=route_name,
        model=model,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        latency_ms=timer.elapsed_ms,
        cost_eur=cost_eur,
        reason=reason,
    )
    return {
        "route": route_name,
        "model": model,
        "answer": _deanonymize(response.text, mapping),
        "latency_ms": timer.elapsed_ms,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "cost_eur": cost_eur,
    }


def _load_existing_results() -> dict[tuple[str, str], dict]:
    if not RESULTS_PATH.exists():
        return {}
    with RESULTS_PATH.open(encoding="utf-8") as f:
        return {(row["question_id"], row["route"]): row for row in csv.DictReader(f)}


def _write_results(results_by_key: dict[tuple[str, str], dict]) -> None:
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(results_by_key.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--routes", nargs="+", choices=["local", "api"], default=["local", "api"]
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="only run the first N questions"
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        default=None,
        help="only run these question_ids (space-separated)",
    )
    args = parser.parse_args()

    with QUESTIONS_PATH.open(encoding="utf-8") as f:
        questions = list(csv.DictReader(f))
    if args.ids:
        wanted = set(args.ids)
        questions = [q for q in questions if q["question_id"] in wanted]
    if args.limit:
        questions = questions[: args.limit]

    # Merge onto whatever's already in RESULTS_PATH (e.g. from a prior --routes
    # api run) instead of clobbering it — each (question_id, route) pair is
    # overwritten by this run only if this run actually produces it, so a
    # slow local run can be restarted or extended without losing prior work.
    results_by_key = _load_existing_results()
    total_attempted = 0

    for q in questions:
        doc_text, doc_count = load_doc_text(q["doc_ids"])
        masked_doc, mapping = anonymize(doc_text)
        decision = route(
            q["question"], masked_doc, had_pii=bool(mapping), doc_count=doc_count
        )
        reason = f"eval run over {q['doc_ids']} — auto-decision was {decision.route} ({decision.reason})"

        for route_name in args.routes:
            total_attempted += 1
            print(f"[{q['question_id']}] route={route_name} ...", end=" ", flush=True)
            try:
                result = run_one_route(
                    route_name, masked_doc, mapping, q["question"], reason
                )
            except Exception as exc:
                print(f"FAILED: {exc}")
                continue
            print(f"{result['latency_ms']:.0f} ms, {result['cost_eur']:.4f} EUR")
            results_by_key[(q["question_id"], route_name)] = {
                "question_id": q["question_id"],
                "complexity": q["complexity"],
                "auto_route": decision.route,
                "auto_reason": decision.reason,
                **result,
            }
            _write_results(
                results_by_key
            )  # flush after every call, not just at the end

    print(
        f"\n{len(results_by_key)} total answers on file "
        f"({total_attempted} attempted this run) written to {RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()
