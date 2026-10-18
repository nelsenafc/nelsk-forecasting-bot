"""The bot's own resolved questions, used to show how similar questions turned out.

Bot makers who looked up similar previously resolved questions did better: 34% of Fall 2025
winners did it, against 0% of non-winners [Metaculus, "AI Forecasting in 2026: What 11
Analyses Say"]. Metaculus hides the outcome of a question from a bot account unless the bot
forecast it, so the lookup draws on the bot's own resolved forecasts. MiniBench repeats its
question templates, so that history grows useful quickly, and it carries the bot's own score
on each one, so a forecaster can see where the bot went wrong before.

The data job keeps history.json on the repo's bot-data branch. The bot reads it from there.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import requests

HISTORY_URL = "https://raw.githubusercontent.com/nelsenafc/nelsk-forecasting-bot/bot-data/history.json"
WORD = re.compile(r"[a-z][a-z'\-]+")
STOP = set(
    "will the of by on in a an be what how many much is are was were to for at as from with and or "
    "than that this these those its it any have has had been before after during between more less "
    "least most per which who whose whom when where would could should does did do not no yes".split()
)
MIN_SIMILARITY = 0.2

_loaded: list[dict] | None = None


def keywords(text: str) -> set[str]:
    return {word for word in WORD.findall(text.lower()) if len(word) > 2 and word not in STOP}


def describe(record: dict) -> str:
    """The bot's final forecast in words, from a scorecard record."""
    final = record.get("final")
    if final is None:
        return "not recorded"
    if isinstance(final, (int, float)):
        return f"{final:.0%}"
    if isinstance(final, dict):
        return " / ".join(f"{option} {p:.0%}" for option, p in final.items())
    low, median, high = final
    return f"median {median:.4g} [80% range {low:.4g} to {high:.4g}]"


def entry(record: dict) -> dict:
    """The fields history keeps from one scorecard record."""
    return {
        "post_id": record["post_id"],
        "question": record["question"],
        "type": record["type"],
        "resolution": record["resolution"],
        "our_forecast": describe(record),
        "peer_score": record.get("peer_score"),
        "resolved_on": record.get("resolved_on"),
    }


def load() -> list[dict]:
    """Resolved history from the bot-data branch, read once per run. Empty if unreachable."""
    global _loaded
    if _loaded is None:
        try:
            response = requests.get(HISTORY_URL, timeout=15)
            response.raise_for_status()
            _loaded = response.json().get("questions", [])
        except (requests.RequestException, ValueError):
            _loaded = []
    return _loaded


def similar(text: str, entries: list[dict], limit: int = 5, exclude: int | None = None) -> list[dict]:
    """Past questions that share the most words with this one, most similar first."""
    wanted = keywords(text)
    if not wanted:
        return []
    scored = []
    for item in entries:
        if item.get("post_id") == exclude:
            continue
        theirs = keywords(item["question"])
        overlap = len(wanted & theirs) / len(wanted | theirs) if theirs else 0.0
        if overlap >= MIN_SIMILARITY:
            scored.append((overlap, item))
    scored.sort(key=lambda pair: -pair[0])
    return [item for _, item in scored[:limit]]


def format_similar(items: list[dict]) -> str:
    if not items:
        return ""
    lines = [
        "These are this bot's own past forecasts on similar questions. Read them as base-rate hints "
        "and as a record of where the bot went wrong, not as answers: wording, dates and thresholds differ."
    ]
    for item in items:
        score = item.get("peer_score")
        verdict = f"; peer score {score:+.0f}" if score is not None else ""
        lines.append(
            f"- {item['question']} [resolved {item.get('resolved_on') or 'date unknown'}]: "
            f"resolved {item['resolution']}; the bot forecast {item['our_forecast']}{verdict}"
        )
    return "\n".join(lines)


def build(records: list[dict]) -> dict:
    """history.json content from the scorecard's resolved records."""
    return {
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "questions": [entry(record) for record in records],
    }


def dumps(content: dict) -> str:
    return json.dumps(content, indent=1, default=str)
