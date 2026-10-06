"""Backtest the price model on history it never saw.

For each series, step through past dates, forecast the value 1-4 weeks ahead from
the data available on that date only, and score the forecast against what happened.
A constant-volatility normal [the usual textbook model, a year of history] is the
baseline.

Reported per series and horizon:
- coverage of the 50%, 80% and 90% central ranges [a calibrated model hits each]
- pinball loss averaged over the 5th-95th percentiles, as a % of the price
  [lower is better; it rewards ranges that are both honest and tight]

Usage: python backtest_price_model.py
"""

from __future__ import annotations

import numpy as np

import price_model

CASES = [
    # source, symbol, kind, bounds, horizons in observations
    ("yahoo", "FXI", "log", None, (5, 10, 20)),
    ("yahoo", "BZ=F", "log", None, (5, 10, 20)),
    ("yahoo", "CL=F", "log", None, (5, 10, 20)),
    ("yahoo", "^GSPC", "log", None, (5, 10, 20)),
    ("yahoo", "GC=F", "log", None, (5, 10, 20)),
    ("yahoo", "EURUSD=X", "log", None, (5, 10, 20)),
    ("yahoo", "BTC-USD", "log", None, (7, 14, 28)),
    ("yahoo", "^VIX", "log", None, (5, 10, 20)),
    ("fred", "GASREGW", "log", None, (1, 2, 4)),
    ("fear_greed", "fng", "level", (0.0, 100.0), (7, 14, 28)),
]
QUANTILES = np.arange(0.05, 0.951, 0.05)
START = 400  # observations of history before the first forecast
STRIDE = 5
PATHS = 2000


def pinball(samples_quantiles: np.ndarray, actual: float) -> float:
    errors = actual - samples_quantiles
    return float(np.mean(np.maximum(QUANTILES * errors, (QUANTILES - 1) * errors)))


def baseline_quantiles(history: np.ndarray, steps: int, kind: str, bounds) -> np.ndarray:
    moves = price_model.changes(history, kind)[-250:]
    sigma = float(np.std(moves)) * np.sqrt(steps)
    rng = np.random.default_rng(0)
    draws = rng.normal(0.0, sigma, size=PATHS)
    terminal = history[-1] * np.exp(draws) if kind == "log" else history[-1] + draws
    if bounds is not None:
        terminal = np.clip(terminal, *bounds)
    return np.quantile(terminal, QUANTILES)


def run_case(source: str, symbol: str, kind: str, bounds, horizons) -> list[dict]:
    series = price_model.fetch(source, symbol)
    values = series.values
    rows = []
    for steps in horizons:
        hits = {0.5: [], 0.8: [], 0.9: []}
        model_loss, base_loss = [], []
        for t in range(START, len(values) - steps, STRIDE):
            history, actual = values[: t + 1], values[t + steps]
            terminal = price_model.simulate_direct(history, steps, kind=kind, n_paths=PATHS, bounds=bounds, seed=t)
            for width in hits:
                low, high = np.quantile(terminal, [(1 - width) / 2, (1 + width) / 2])
                hits[width].append(low <= actual <= high)
            scale = history[-1] if kind == "log" else 100.0
            model_loss.append(pinball(np.quantile(terminal, QUANTILES), actual) / scale)
            base_loss.append(pinball(baseline_quantiles(history, steps, kind, bounds), actual) / scale)
        rows.append(
            {
                "series": symbol,
                "steps": steps,
                "n": len(model_loss),
                "cov50": float(np.mean(hits[0.5])),
                "cov80": float(np.mean(hits[0.8])),
                "cov90": float(np.mean(hits[0.9])),
                "loss": float(np.mean(model_loss)) * 100,
                "base": float(np.mean(base_loss)) * 100,
            }
        )
    return rows


def main() -> None:
    print(f"{'series':<10}{'steps':>6}{'n':>6}{'50%':>7}{'80%':>7}{'90%':>7}{'loss':>9}{'base':>9}{'vs base':>9}")
    all_rows = []
    for case in CASES:
        try:
            rows = run_case(*case)
        except Exception as error:  # one bad feed shouldn't stop the report
            print(f"{case[1]:<10} skipped: {error}")
            continue
        for row in rows:
            all_rows.append(row)
            print(
                f"{row['series']:<10}{row['steps']:>6}{row['n']:>6}{row['cov50']:>7.0%}{row['cov80']:>7.0%}"
                f"{row['cov90']:>7.0%}{row['loss']:>9.3f}{row['base']:>9.3f}{row['loss'] / row['base'] - 1:>+9.1%}"
            )
    if all_rows:
        print(
            f"\nAverage coverage - 50%: {np.mean([r['cov50'] for r in all_rows]):.0%}, "
            f"80%: {np.mean([r['cov80'] for r in all_rows]):.0%}, 90%: {np.mean([r['cov90'] for r in all_rows]):.0%}. "
            f"Loss vs baseline: {np.mean([r['loss'] / r['base'] - 1 for r in all_rows]):+.1%}"
        )


if __name__ == "__main__":
    main()
