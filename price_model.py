"""Statistical forecasts for questions about where a market series will be on a date.

For a question like "What will Brent settle at on 16 Oct?", the bot pulls the
series' history and simulates where it lands on the target date with filtered
historical simulation:
- every past window as long as the forecast horizon is taken whole, and its total
  change divided by the volatility known when it started [an EWMA],
- those standardised changes are resampled at random,
- and rescaled to today's volatility.
That keeps the fat tails real markets have, carries any tendency to trend or pull
back over that horizon, and sizes the spread to the current regime instead of a
long-run average.

Prices move in log space [kind "log"]. Bounded indices such as the crypto Fear
& Greed index move in plain points [kind "level"] and are clipped to their bounds.
"""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import numpy as np
import requests

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 20
EWMA_LAMBDA = 0.94  # RiskMetrics' daily decay: roughly a 2-3 week memory
MIN_HISTORY = 250  # about a year of daily observations


@dataclass
class Series:
    source: str
    symbol: str
    dates: list[date]
    values: np.ndarray

    @property
    def last_date(self) -> date:
        return self.dates[-1]

    @property
    def last_value(self) -> float:
        return float(self.values[-1])


def fetch_yahoo(symbol: str) -> Series:
    response = requests.get(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
        params={"range": "5y", "interval": "1d"},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    result = response.json()["chart"]["result"][0]
    closes = result["indicators"]["quote"][0]["close"]
    points = [
        (datetime.fromtimestamp(stamp, tz=timezone.utc).date(), float(close))
        for stamp, close in zip(result["timestamp"], closes)
        if close is not None
    ]
    return _series("yahoo", symbol, points)


def fetch_fred(series_id: str) -> Series:
    response = requests.get(
        "https://fred.stlouisfed.org/graph/fredgraph.csv",
        params={"id": series_id},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    points = []
    for row in csv.DictReader(io.StringIO(response.text)):
        value = row.get(series_id)
        if value in (None, "", "."):
            continue
        points.append((date.fromisoformat(row["observation_date"]), float(value)))
    return _series("fred", series_id, points)


def fetch_fear_greed() -> Series:
    response = requests.get(
        "https://api.alternative.me/fng/", params={"limit": 0}, timeout=REQUEST_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    points = [
        (datetime.fromtimestamp(int(row["timestamp"]), tz=timezone.utc).date(), float(row["value"]))
        for row in response.json()["data"]
    ]
    return _series("fear_greed", "crypto fear and greed index", points)


def fetch(source: str, symbol: str) -> Series:
    if source == "yahoo":
        return fetch_yahoo(symbol)
    if source == "fred":
        return fetch_fred(symbol)
    if source == "fear_greed":
        return fetch_fear_greed()
    raise ValueError(f"Unknown data source: {source}")


def _series(source: str, symbol: str, points: list[tuple[date, float]]) -> Series:
    by_date = dict(sorted(points))  # one value per date, latest wins
    dates = list(by_date)
    return Series(source, symbol, dates, np.array([by_date[d] for d in dates], dtype=float))


def changes(values: np.ndarray, kind: str) -> np.ndarray:
    if kind == "log":
        if np.any(values <= 0):
            raise ValueError("Log changes need strictly positive values")
        return np.diff(np.log(values))
    return np.diff(values)


def ewma_volatility(moves: np.ndarray, decay: float = EWMA_LAMBDA) -> np.ndarray:
    """sigma[t] is the volatility known before move t; sigma[-1] covers the next move."""
    if len(moves) < 30:
        raise ValueError("Need at least 30 moves to estimate volatility")
    variance = float(np.mean(moves[:30] ** 2)) or 1e-12
    sigmas = np.empty(len(moves) + 1)
    for index, move in enumerate(moves):
        sigmas[index] = np.sqrt(variance)
        variance = decay * variance + (1 - decay) * move**2
    sigmas[-1] = np.sqrt(variance)
    return np.maximum(sigmas, 1e-12)


def frequency(dates: list[date]) -> str:
    """'daily' [includes weekends], 'business' [weekdays] or 'weekly', from recent spacing."""
    recent = dates[-60:]
    gaps = [(later - earlier).days for earlier, later in zip(recent, recent[1:])]
    if gaps and float(np.median(gaps)) >= 5:
        return "weekly"
    if any(day.weekday() >= 5 for day in recent):
        return "daily"
    return "business"


def steps_until(last: date, target: date, freq: str) -> int:
    """Observations between the last one we have and the target date, inclusive of the target."""
    if target <= last:
        return 0
    if freq == "weekly":
        return int((target - last).days // 7)
    if freq == "daily":
        return (target - last).days
    return int(np.busday_count(last + timedelta(days=1), target + timedelta(days=1)))


def simulate(
    values: np.ndarray,
    steps: int,
    kind: str = "log",
    n_paths: int = 20000,
    bounds: tuple[float, float] | None = None,
    seed: int | None = 0,
) -> np.ndarray:
    """Terminal values of n_paths simulated paths, `steps` observations ahead."""
    if steps < 1:
        raise ValueError("Need at least one step to simulate")
    moves = changes(values, kind)
    sigmas = ewma_volatility(moves)
    standardised = moves / sigmas[:-1]
    standardised = standardised - standardised.mean()
    rng = np.random.default_rng(seed)
    draws = rng.choice(standardised, size=(n_paths, steps), replace=True)
    total = draws.sum(axis=1) * sigmas[-1]
    if kind == "log":
        terminal = values[-1] * np.exp(total)
    else:
        terminal = values[-1] + total
    if bounds is not None:
        terminal = np.clip(terminal, bounds[0], bounds[1])
    return terminal


def simulate_direct(
    values: np.ndarray,
    steps: int,
    kind: str = "log",
    n_paths: int = 20000,
    bounds: tuple[float, float] | None = None,
    seed: int | None = 0,
) -> np.ndarray:
    """Resample whole past `steps`-long changes instead of single days.

    Each historical window's total change is divided by the volatility known at
    its start, then rescaled to today's. Because windows are taken whole, any
    tendency to trend [gas prices] or to pull back [VIX, sentiment indices] over
    that horizon carries into the forecast, which single-day resampling loses.
    """
    if steps < 1:
        raise ValueError("Need at least one step to simulate")
    level = np.log(values) if kind == "log" else values.astype(float)
    sigmas = ewma_volatility(np.diff(level))
    window_changes = level[steps:] - level[:-steps]
    standardised = window_changes / sigmas[: len(window_changes)]
    standardised = standardised - standardised.mean()
    rng = np.random.default_rng(seed)
    total = rng.choice(standardised, size=n_paths, replace=True) * sigmas[-1]
    terminal = values[-1] * np.exp(total) if kind == "log" else values[-1] + total
    if bounds is not None:
        terminal = np.clip(terminal, bounds[0], bounds[1])
    return terminal


def daily_volatility(values: np.ndarray, kind: str = "log") -> float:
    return float(ewma_volatility(changes(values, kind))[-1])


@dataclass
class PriceForecast:
    series: Series
    target: date
    steps: int
    freq: str
    kind: str
    multiplier: float
    terminal: np.ndarray  # simulated values in the question's units

    def percentile(self, q: float) -> float:
        return float(np.quantile(self.terminal, q))

    def summary(self) -> str:
        unit_note = "" if self.multiplier == 1 else f" [series value x {self.multiplier:g} to match the question's units]"
        vol = daily_volatility(self.series.values, self.kind)
        vol_text = f"{vol:.2%} per day" if self.kind == "log" else f"{vol:.2f} points per day"
        step_name = {"weekly": "weekly", "daily": "daily", "business": "trading-day"}[self.freq]
        quantiles = ", ".join(
            f"{int(q * 100)}%: {self.percentile(q):,.4g}" for q in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
        )
        return (
            f"Statistical model from {self.series.source} series '{self.series.symbol}'{unit_note}. "
            f"Last observation {self.series.last_value * self.multiplier:,.4g} on {self.series.last_date.isoformat()}. "
            f"Current volatility {vol_text}. {self.steps} {step_name} steps to {self.target.isoformat()}. "
            f"Simulated distribution percentiles - {quantiles}. "
            "This assumes no drift and today's volatility; it knows nothing about scheduled events or news."
        )


def forecast(
    source: str,
    symbol: str,
    target: date,
    kind: str = "log",
    multiplier: float = 1.0,
    bounds: tuple[float, float] | None = None,
    n_paths: int = 20000,
) -> PriceForecast | None:
    """Simulate the series out to the target date; None if the data can't support it."""
    series = fetch(source, symbol)
    if len(series.values) < MIN_HISTORY:
        logger.warning(f"Only {len(series.values)} observations for {symbol}; skipping the price model")
        return None
    freq = frequency(series.dates)
    steps = steps_until(series.last_date, target, freq)
    if steps < 1:
        logger.warning(f"Target {target} is not after the last observation of {symbol}; skipping")
        return None
    # Whole-window resampling: it kept 80% ranges at 80% coverage across ten series in
    # backtest_price_model.py, where single-day resampling ran too wide on indices that
    # pull back to their average and too narrow on trending ones like gas prices.
    terminal = simulate_direct(series.values, steps, kind=kind, n_paths=n_paths, bounds=bounds) * multiplier
    return PriceForecast(series, target, steps, freq, kind, multiplier, terminal)
