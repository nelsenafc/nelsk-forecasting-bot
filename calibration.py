"""Learn a correction for the bot's yes/no forecasts from its own resolved ones, and test caps.

Platt scaling [logistic recalibration] maps p to sigmoid(a * logit(p) + b). Metaculus found it
the best of five post-hoc corrections for bot forecasts, though the gain was small [Brier
-0.016 on binary questions]. Capping forecasts at a floor and ceiling was the strongest
difference between winning bots in Fall 2025. Source: "AI Forecasting in 2026: What 11
Analyses Say".

Both are judged here only on questions they were not fitted on [leave one out], and go live
only through calibration.json, once that test is positive over MIN_QUESTIONS or more. A guessed
correction is worse than none: pushing forecasts away from 50% by a factor of 1.3 would have
cost the bot 34.5 points on its first 7 resolved yes/no questions.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from ensemble import BINARY_CAP

MIN_QUESTIONS = 50
CLIP = (0.01, 0.99)  # the bot's limits before the 5%/95% cap went live on 18 Oct 2026; the test's baseline
CAPS = [(0.05, 0.95), (0.10, 0.90), (0.15, 0.85)]  # the live cap first, then tighter candidates
CONFIG = Path(__file__).with_name("calibration.json")


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def fit(pairs: list[tuple[float, bool]], pull: float = 2.0) -> tuple[float, float]:
    """Maximum-likelihood a and b, pulled toward no change [a=1, b=0] so a few questions can't swing them."""
    a, b = 1.0, 0.0
    for _ in range(50):
        grad_a = grad_b = h_aa = h_ab = h_bb = 0.0
        for p, happened in pairs:
            x = _logit(p)
            q = _sigmoid(a * x + b)
            residual, weight = (1.0 if happened else 0.0) - q, q * (1 - q)
            grad_a += residual * x
            grad_b += residual
            h_aa += weight * x * x
            h_ab += weight * x
            h_bb += weight
        grad_a -= pull * (a - 1)
        grad_b -= pull * b
        h_aa += pull
        h_bb += pull
        det = h_aa * h_bb - h_ab * h_ab
        if det <= 0:
            break
        step_a = (h_bb * grad_a - h_ab * grad_b) / det
        step_b = (h_aa * grad_b - h_ab * grad_a) / det
        a, b = a + step_a, b + step_b
        if abs(step_a) + abs(step_b) < 1e-9:
            break
    return a, b


def apply(p: float, a: float = 1.0, b: float = 0.0, floor: float = CLIP[0], ceiling: float = CLIP[1]) -> float:
    return min(max(_sigmoid(a * _logit(p) + b), floor), ceiling)


def log_score(p: float, happened: bool) -> float:
    """100 x natural log of the probability given to what happened, the scale peer scores use."""
    return 100 * math.log(p if happened else 1 - p)


def _raw(p: float) -> float:
    return min(max(p, CLIP[0]), CLIP[1])


def leave_one_out(pairs: list[tuple[float, bool]]) -> list[float]:
    """Score change on each question when the correction is fitted on all the others."""
    gains = []
    for i, (p, happened) in enumerate(pairs):
        a, b = fit(pairs[:i] + pairs[i + 1 :])
        gains.append(log_score(apply(p, a, b), happened) - log_score(_raw(p), happened))
    return gains


def cap_gains(pairs: list[tuple[float, bool]], floor: float, ceiling: float) -> list[float]:
    """Score change on each question from capping at floor and ceiling. Caps are fixed rules, nothing is fitted."""
    return [
        log_score(min(max(p, floor), ceiling), happened) - log_score(_raw(p), happened) for p, happened in pairs
    ]


def load() -> dict:
    try:
        return json.loads(CONFIG.read_text())
    except (OSError, ValueError):
        return {}


def apply_binary(p: float) -> float:
    """The live correction from calibration.json, never looser than the safety cap; the forecast unchanged when there is none."""
    config = load().get("binary")
    if not config:
        return p
    return apply(
        p,
        config.get("a", 1.0),
        config.get("b", 0.0),
        max(config.get("floor", BINARY_CAP[0]), BINARY_CAP[0]),
        min(config.get("ceiling", BINARY_CAP[1]), BINARY_CAP[1]),
    )
