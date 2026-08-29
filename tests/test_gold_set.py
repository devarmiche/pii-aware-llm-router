import json
from pathlib import Path

import pytest

GOLD_SET = Path(__file__).resolve().parents[1] / "data/synthetic/gold_set.jsonl"


@pytest.fixture(scope="module")
def docs() -> list[dict]:
    """Parse the gold set"""
    assert GOLD_SET.exists(), "gold_set.jsonl not found"
    with GOLD_SET.open() as f:
        return [json.loads(line) for line in f]


def test_spans_match_text(docs: list[dict]) -> None:
    for doc in docs:
        for entity in doc["entities"]:
            extracted = doc["text"][entity["start"] : entity["end"]]
            assert extracted == entity["value"], (
                f"{doc['doc_id']}:{entity['type']} span"
                f"[{entity['start']}:{entity['end']}] yields {extracted!r},"
                f"annotated as {entity['value']!r}"
            )
