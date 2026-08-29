import json
from collections import Counter
from itertools import pairwise
from pathlib import Path

import pytest

GOLD_SET = Path(__file__).resolve().parents[1] / "data/synthetic/gold_set.jsonl"


@pytest.fixture(scope="module")
def docs() -> list[dict]:
    """Parse the gold set"""
    assert GOLD_SET.exists(), "gold_set.jsonl not found"
    with GOLD_SET.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def test_spans_match_text(docs: list[dict]) -> None:
    for doc in docs:
        for entity in doc["entities"]:
            extracted = doc["text"][entity["start"] : entity["end"]]
            assert extracted == entity["value"], (
                f"{doc['doc_id']}:{entity['type']} span "
                f"[{entity['start']}:{entity['end']}] yields {extracted!r}, "
                f"annotated as {entity['value']!r}"
            )


def test_spans_do_not_overlap(docs: list[dict]) -> None:
    """Overlapping ground truth makes a detection impossible to score."""
    for doc in docs:
        ordered = sorted(doc["entities"], key=lambda e: e["start"])
        for previous, current in pairwise(ordered):
            assert current["start"] >= previous["end"], (
                f"{doc['doc_id']}: {previous['type']} "
                f"[{previous['start']}:{previous['end']}] overlaps {current['type']} "
                f"[{current['start']}:{current['end']}]"
            )


def test_entity_composition_is_stable_per_template(docs: list[dict]) -> None:
    """Every document from a template carries the same entity types.

    Compared against the first document of each template rather than a
    hardcoded total, so adding templates does not require editing this test.
    """
    reference: dict[str, Counter] = {}
    for doc in docs:
        template = doc["meta"]["template"]
        composition = Counter(entity["type"] for entity in doc["entities"])
        expected = reference.setdefault(template, composition)
        assert composition == expected, (
            f"{doc['doc_id']}: composition {dict(composition)} differs from "
            f"the first {template} document {dict(expected)}"
        )


def test_schema_is_complete(docs: list[dict]) -> None:
    """Fields every downstream consumer relies on, and unique document ids."""
    assert docs, "gold set is empty"

    seen: set[str] = set()
    for doc in docs:
        assert {"doc_id", "text", "entities", "meta"} <= doc.keys(), (
            f"missing top-level keys: {doc.keys()}"
        )
        assert "template" in doc["meta"], f"{doc['doc_id']}: meta.template missing"
        assert doc["doc_id"] not in seen, f"duplicate doc_id: {doc['doc_id']}"
        seen.add(doc["doc_id"])

        for entity in doc["entities"]:
            assert {"type", "start", "end", "value"} <= entity.keys(), (
                f"{doc['doc_id']}: incomplete entity {entity}"
            )
            assert entity["type"], f"{doc['doc_id']}: entity with empty type"
            assert entity["start"] < entity["end"], (
                f"{doc['doc_id']}: non-positive span on {entity['type']}"
            )
