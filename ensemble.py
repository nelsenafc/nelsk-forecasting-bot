"""Who forecasts, and how their forecasts are combined.

Each AI lab gets equal weight in the final answer however many samples it
contributes, so one lab's house style can't dominate. Binary forecasts are
pooled in log-odds space, multiple choice as a weighted average, and numeric
or date forecasts as a weighted average of their CDFs [a mixture].
"""

from __future__ import annotations

import math
import os
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class ForecasterSpec:
    lab: str
    model: str
    samples: int = 1
    reasoning_effort: str | None = "high"
    timeout: int = 420

    @property
    def short_name(self) -> str:
        return self.model.removeprefix("openrouter/")


def default_forecasters() -> list[ForecasterSpec]:
    """The v1 line-up. Each model and sample count can be overridden with an env var."""
    return [
        ForecasterSpec(
            lab="openai",
            model=os.getenv("FORECASTER_OPENAI_MODEL", "openrouter/openai/gpt-6.1-sol"),
            samples=int(os.getenv("FORECASTER_OPENAI_SAMPLES", "2")),
        ),
        ForecasterSpec(
            lab="anthropic",
            model=os.getenv(
                "FORECASTER_ANTHROPIC_MODEL", "openrouter/anthropic/claude-sonnet-5.5"
            ),
            samples=int(os.getenv("FORECASTER_ANTHROPIC_SAMPLES", "1")),
            # Claude's thinking budget scales with effort; medium keeps a call near $0.10.
            reasoning_effort=os.getenv("FORECASTER_ANTHROPIC_EFFORT", "medium"),
        ),
        ForecasterSpec(
            lab="google",
            model=os.getenv(
                "FORECASTER_GOOGLE_MODEL", "openrouter/google/gemini-3.8-flash"
            ),
            samples=int(os.getenv("FORECASTER_GOOGLE_SAMPLES", "2")),
        ),
    ]


def lab_weights(labs: list[str]) -> list[float]:
    """Weight per sample so that every lab's samples sum to the same total."""
    if not labs:
        raise ValueError("Need at least one forecast to weight")
    counts = Counter(labs)
    return [1.0 / (len(counts) * counts[lab]) for lab in labs]


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def combine_binary(
    probabilities: list[float],
    labs: list[str],
    sample_floor: float = 0.005,
    clip: tuple[float, float] = (0.01, 0.99),
) -> float:
    if len(probabilities) != len(labs):
        raise ValueError("Each probability needs a lab")
    weights = lab_weights(labs)
    pooled = sum(
        weight * logit(min(max(p, sample_floor), 1.0 - sample_floor))
        for p, weight in zip(probabilities, weights)
    )
    return min(max(sigmoid(pooled), clip[0]), clip[1])


def combine_multiple_choice(
    distributions: list[dict[str, float]],
    labs: list[str],
    floor: float = 0.005,
) -> dict[str, float]:
    if len(distributions) != len(labs):
        raise ValueError("Each distribution needs a lab")
    if not distributions:
        raise ValueError("Need at least one distribution")
    options = list(distributions[0])
    for distribution in distributions:
        if set(distribution) != set(options):
            raise ValueError(
                f"Option names differ between forecasts: {sorted(distribution)} vs {sorted(options)}"
            )
    weights = lab_weights(labs)
    pooled = {option: 0.0 for option in options}
    for distribution, weight in zip(distributions, weights):
        total = sum(distribution.values())
        if total <= 0:
            raise ValueError("A forecast put zero probability on every option")
        for option in options:
            pooled[option] += weight * distribution[option] / total
    floored = {option: max(p, floor) for option, p in pooled.items()}
    total = sum(floored.values())
    return {option: p / total for option, p in floored.items()}


def combine_cdfs(cdfs: list[list[float]], labs: list[str]) -> list[float]:
    """Pointwise weighted mean of CDF heights on a shared x-axis."""
    if len(cdfs) != len(labs):
        raise ValueError("Each CDF needs a lab")
    if not cdfs:
        raise ValueError("Need at least one CDF")
    length = len(cdfs[0])
    if any(len(cdf) != length for cdf in cdfs):
        raise ValueError("CDFs must share the same x-axis")
    weights = lab_weights(labs)
    return [
        sum(weight * cdf[i] for cdf, weight in zip(cdfs, weights))
        for i in range(length)
    ]
