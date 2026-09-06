"""Measures the anonymizer against data/synthetic/gold_set.jsonl.

Matching is overlap-based, not exact-boundary: a predicted span counts as a
hit for a gold entity if they share the same type and their character ranges
intersect at all (see project brief for why — exact matching undercounts
partial detections like a NER model tagging "rue Morin" inside a full
address span). This can double-count in edge cases (one prediction
overlapping two adjacent gold entities); acceptable for this project's scope.
"""

import json
from collections import defaultdict
from pathlib import Path

from presidio_analyzer import RecognizerResult

from src.anonymizer import detect

GOLD_SET = Path("data/synthetic/gold_set.jsonl")

Run = tuple[dict, list[RecognizerResult]]


def _load_docs() -> list[dict]:
    with GOLD_SET.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _run_detection(docs: list[dict]) -> list[Run]:
    """Run detect() once per doc; every metric below reuses this."""
    return [(doc, detect(doc["text"])) for doc in docs]


def per_type_metrics(runs: list[Run]) -> tuple[dict[str, dict[str, int]], int]:
    known_types = {e["type"] for doc, _ in runs for e in doc["entities"]}
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"gold": 0, "matched": 0, "predicted": 0, "true_pred": 0}
    )
    spurious = 0

    for doc, predicted in runs:
        gold = doc["entities"]

        for g in gold:
            counts[g["type"]]["gold"] += 1
            hit = any(
                p.entity_type == g["type"] and _overlaps(g["start"], g["end"], p.start, p.end)
                for p in predicted
            )
            if hit:
                counts[g["type"]]["matched"] += 1

        for p in predicted:
            if p.entity_type not in known_types:
                spurious += 1
                continue
            counts[p.entity_type]["predicted"] += 1
            hit = any(
                g["type"] == p.entity_type and _overlaps(g["start"], g["end"], p.start, p.end)
                for g in gold
            )
            if hit:
                counts[p.entity_type]["true_pred"] += 1

    return counts, spurious


def person_recall_by_origin(runs: list[Run]) -> dict[str, tuple[int, int]]:
    hits: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)

    for doc, predicted in runs:
        for g in doc["entities"]:
            if g["type"] != "PERSON":
                continue
            origin = g["name_origin"]
            total[origin] += 1
            if any(
                p.entity_type == "PERSON" and _overlaps(g["start"], g["end"], p.start, p.end)
                for p in predicted
            ):
                hits[origin] += 1

    return {origin: (hits[origin], total[origin]) for origin in total}


def print_report(
    docs: list[dict],
    counts: dict[str, dict[str, int]],
    spurious: int,
    origin_recall: dict[str, tuple[int, int]],
) -> None:
    n_entities = sum(len(d["entities"]) for d in docs)
    print(f"Gold set: {len(docs)} documents, {n_entities} entities\n")

    header = f"{'TYPE':<15}{'GOLD':>6}{'PRED':>6}{'PRECISION':>11}{'RECALL':>9}{'F1':>7}"
    print(header)
    print("-" * len(header))
    for entity_type in sorted(counts):
        c = counts[entity_type]
        precision = _safe_div(c["true_pred"], c["predicted"])
        recall = _safe_div(c["matched"], c["gold"])
        f1 = _safe_div(2 * precision * recall, precision + recall)
        print(
            f"{entity_type:<15}{c['gold']:>6}{c['predicted']:>6}"
            f"{precision:>11.2f}{recall:>9.2f}{f1:>7.2f}"
        )

    print(f"\nSpurious detections (type outside gold taxonomy, e.g. ORGANIZATION/URL): {spurious}")

    print("\nPERSON recall by name origin:")
    for origin, (hits, total) in sorted(origin_recall.items()):
        print(f"  {origin:<15}{hits}/{total} ({_safe_div(hits, total):.2f})")


def main() -> None:
    docs = _load_docs()
    runs = _run_detection(docs)
    counts, spurious = per_type_metrics(runs)
    origin_recall = person_recall_by_origin(runs)
    print_report(docs, counts, spurious, origin_recall)


if __name__ == "__main__":
    main()
