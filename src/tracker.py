import csv
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Literal

LOG_PATH = Path("data/tracker_log.csv")
FIELDNAMES = [
    "timestamp",
    "route",
    "model",
    "input_tokens",
    "output_tokens",
    "latency_ms",
    "ttft_ms",
    "cost_eur",
    "reason",
]

# model -> (price per 1k input tokens, price per 1k output tokens), in EUR.
# Source: OpenRouter's public /api/v1/models pricing (USD per token), fetched
# 2026-09-19: mistralai/mistral-large-2407 at $0.000002 in / $0.000006 out
# (this is also what the floating "mistralai/mistral-large" alias resolves to
# — the newer "-2512" snapshot only exists as a :batch/async variant on
# OpenRouter and can't be called synchronously, so it's not used here).
# Converted at 1 USD = 0.8726 EUR (ECB reference rate via frankfurter.dev,
# 2026-09-18). Re-derive both the per-token price and the exchange rate
# before trusting this for a new date.
PRICE_EUR_PER_1K_TOKENS: dict[str, tuple[float, float]] = {
    "mistralai/mistral-large-2407": (0.001745, 0.005236),
}


class Timer:
    """with Timer() as t: ... ; t.elapsed_ms"""

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        self.elapsed_ms: float | None = None
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000


def compute_api_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    if model not in PRICE_EUR_PER_1K_TOKENS:
        raise ValueError(
            f"no pricing registered for {model!r} — add it to "
            "PRICE_EUR_PER_1K_TOKENS before logging a real API call"
        )
    price_in, price_out = PRICE_EUR_PER_1K_TOKENS[model]
    return (input_tokens / 1000) * price_in + (output_tokens / 1000) * price_out


def log_call(
    route: Literal["local", "api"],
    model: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
    cost_eur: float,
    reason: str,
    ttft_ms: float | None = None,
) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new = not LOG_PATH.exists()
    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if is_new:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": time.time(),
                "route": route,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": latency_ms,
                "ttft_ms": "" if ttft_ms is None else ttft_ms,
                "cost_eur": cost_eur,
                "reason": reason,
            }
        )


def _read_rows() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    with LOG_PATH.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize() -> None:
    rows = _read_rows()
    if not rows:
        print("No calls logged yet.")
        return

    by_route: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_route[row["route"]].append(row)

    total_cost = sum(float(r["cost_eur"]) for r in rows)
    print(f"{len(rows)} calls logged, total cost: {total_cost:.4f} EUR\n")

    header = f"{'ROUTE':<8}{'CALLS':>7}{'SHARE':>8}{'AVG LATENCY (ms)':>20}{'AVG COST (EUR)':>16}"
    print(header)
    print("-" * len(header))
    for route_name, group in sorted(by_route.items()):
        share = len(group) / len(rows)
        avg_latency = mean(float(r["latency_ms"]) for r in group)
        avg_cost = mean(float(r["cost_eur"]) for r in group)
        print(
            f"{route_name:<8}{len(group):>7}{share:>8.0%}"
            f"{avg_latency:>20.1f}{avg_cost:>16.4f}"
        )


if __name__ == "__main__":
    summarize()
