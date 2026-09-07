import time
from pathlib import Path

import pytest

from src import tracker


@pytest.fixture(autouse=True)
def _isolated_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect LOG_PATH so tests never touch the real tracker_log.csv,
    which is meant to hold real measurements only (see claude.md)."""
    monkeypatch.setattr(tracker, "LOG_PATH", tmp_path / "tracker_log.csv")


def test_timer_measures_elapsed_time() -> None:
    with tracker.Timer() as t:
        time.sleep(0.01)
    assert t.elapsed_ms is not None
    assert t.elapsed_ms >= 10


def test_compute_api_cost_uses_registered_pricing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(tracker.PRICE_EUR_PER_1K_TOKENS, "test-model", (0.002, 0.006))
    cost = tracker.compute_api_cost("test-model", input_tokens=1000, output_tokens=500)
    assert cost == pytest.approx(0.002 + 0.003)


def test_compute_api_cost_raises_for_unregistered_model() -> None:
    with pytest.raises(ValueError, match="no pricing registered"):
        tracker.compute_api_cost("unregistered-model", input_tokens=10, output_tokens=10)


def test_log_call_creates_file_with_header() -> None:
    assert not tracker.LOG_PATH.exists()
    tracker.log_call("local", "mistral:7b-instruct", 100, 50, 500.0, 0.0, "reason")
    assert tracker.LOG_PATH.exists()
    rows = tracker._read_rows()
    assert len(rows) == 1
    assert rows[0]["route"] == "local"
    assert rows[0]["model"] == "mistral:7b-instruct"
    assert rows[0]["ttft_ms"] == ""


def test_log_call_appends_multiple_rows() -> None:
    tracker.log_call("local", "mistral:7b-instruct", 100, 50, 500.0, 0.0, "reason a")
    tracker.log_call("api", "mistral-large", 200, 100, 1200.0, 0.01, "reason b", ttft_ms=150.0)
    rows = tracker._read_rows()
    assert len(rows) == 2
    assert rows[1]["route"] == "api"
    assert rows[1]["ttft_ms"] == "150.0"


def test_summarize_with_no_calls(capsys: pytest.CaptureFixture[str]) -> None:
    tracker.summarize()
    assert "No calls logged yet." in capsys.readouterr().out


def test_summarize_computes_route_split_and_averages(capsys: pytest.CaptureFixture[str]) -> None:
    tracker.log_call("local", "mistral:7b-instruct", 100, 50, 500.0, 0.0, "reason")
    tracker.log_call("local", "mistral:7b-instruct", 100, 50, 700.0, 0.0, "reason")
    tracker.log_call("api", "mistral-large", 200, 100, 1200.0, 0.01, "reason")

    tracker.summarize()
    out = capsys.readouterr().out

    assert "3 calls logged" in out
    assert "local" in out
    assert "api" in out
