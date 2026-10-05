"""Export the bot's tournament forecasts as rows for a human-readable forecast log.

Forecast values are exported only once a question has closed. While a question is
still open its row carries the public facts only: tournament rules forbid a human
seeing the bot's forecast before close, and this log is read by humans.

Usage: python forecast_log.py > forecast_log.json
"""

from __future__ import annotations

import json
import os
import re
import sys
import time

import dotenv

dotenv.load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

import requests
from forecasting_tools import (
    BinaryQuestion,
    DateQuestion,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.data_models.questions import DiscreteQuestion, QuestionState

from forecaster import _pct, compact_prediction
from guards import may_preview

API = "https://www.metaculus.com/api"
TYPE_NAMES = [
    (DiscreteQuestion, "Count"),
    (NumericQuestion, "Numeric"),
    (DateQuestion, "Date"),
    (MultipleChoiceQuestion, "Multiple choice"),
    (BinaryQuestion, "Binary"),
]
STATUS_NAMES = {QuestionState.RESOLVED: "Resolved", QuestionState.CLOSED: "Closed"}
SUMMARY_SECTION = re.compile(r"#+ How the final forecast was made\s*\n(.*?)(?:\n#+ |\Z)", re.S)
MODEL_LINE = re.compile(r"^- ([\w.\-]+/[\w.\-]+): (.+)$", re.M)


def type_name(question: MetaculusQuestion) -> str:
    for cls, name in TYPE_NAMES:
        if isinstance(question, cls):
            return name
    return type(question).__name__


def tournament_name(question: MetaculusQuestion) -> str | None:
    slugs = " ".join(question.tournament_slugs or []).lower()
    if "minibench" in slugs:
        return "MiniBench"
    if "futureeval" in slugs or question.default_project_id == MetaculusClient.FE_FALL_2026_ID:
        return "Fall 2026"
    return None


def model_summary(comment: str) -> str:
    """Per-model forecasts from the 'How the final forecast was made' block of the bot's comment."""
    section = SUMMARY_SECTION.search(comment or "")
    if not section:
        return ""
    parts = [
        f"{model.split('/')[-1]} {value.strip()}"
        for model, value in MODEL_LINE.findall(section.group(1))
        if value.strip() and not value.strip().endswith(":")
    ]
    return "; ".join(parts)


def final_forecast(question: MetaculusQuestion) -> str:
    if isinstance(question, MultipleChoiceQuestion):
        latest = (question.api_json.get("question", {}).get("my_forecasts") or {}).get("latest") or {}
        values = latest.get("forecast_values") or []
        return " / ".join(f"{option} {_pct(p)}" for option, p in zip(question.options, values))
    if not question.previous_forecasts:
        return ""
    last = question.previous_forecasts[-1]
    if isinstance(question, BinaryQuestion):
        return _pct(last.prediction_in_decimal)
    return compact_prediction(last)


def score_of(question: MetaculusQuestion) -> float | None:
    data = (question.api_json.get("question", {}).get("my_forecasts") or {}).get("score_data") or {}
    for key in ("spot_peer_score", "peer_score"):
        if data.get(key) is not None:
            return round(float(data[key]), 1)
    return None


def build_row(question: MetaculusQuestion, comment: str, tournament: str) -> dict:
    visible = may_preview(question)
    resolution = question.api_json.get("question", {}).get("resolution")
    return {
        "question": question.question_text,
        "url": question.page_url,
        "tournament": tournament,
        "type": type_name(question),
        "status": STATUS_NAMES.get(question.state, "Open"),
        "closed": question.close_time.date().isoformat() if question.close_time else None,
        "resolves": (
            question.scheduled_resolution_time.date().isoformat()
            if question.scheduled_resolution_time
            else None
        ),
        "forecast": final_forecast(question) if visible else "",
        "models": model_summary(comment) if visible else "",
        "outcome": str(resolution) if resolution is not None else "",
        "peer_score": score_of(question),
    }


def bot_comments(headers: dict, bot_id: int) -> dict[int, str]:
    """Latest comment text per post, private and public, fetching archived comments in full."""
    texts: dict[int, str] = {}
    page_size = 100
    for private in ("true", "false"):
        offset = 0
        while True:
            # The API always returns a "next" link, even past the last result, so page by
            # offset and stop on a short page rather than following "next".
            response = requests.get(
                f"{API}/comments/",
                headers=headers,
                params={"author": bot_id, "is_private": private, "limit": page_size, "offset": offset},
                timeout=30,
            )
            response.raise_for_status()
            results = response.json().get("results", [])
            for comment in results:
                author = comment.get("author")
                author_id = author.get("id") if isinstance(author, dict) else author
                post = comment.get("on_post")
                if author_id != bot_id or post in texts:
                    continue
                text = comment.get("text") or ""
                if comment.get("is_text_archived"):
                    time.sleep(1.3)  # the full-text endpoint allows 8 calls per 10 seconds
                    full = requests.get(f"{API}/comments/{comment['id']}/", headers=headers, timeout=30)
                    if full.ok:
                        text = full.json().get("text") or text
                texts[post] = text
            if len(results) < page_size:
                break
            offset += page_size
    return texts


def main() -> None:
    client = MetaculusClient()
    headers = {"Authorization": f"Token {os.environ['METACULUS_TOKEN']}"}
    rows = []
    for post_id, comment in bot_comments(headers, client.get_current_user_id()).items():
        question = client.get_question_by_post_id(post_id)
        tournament = tournament_name(question)
        if tournament is None:  # practice and main-site questions stay out of the log
            continue
        rows.append(build_row(question, comment, tournament))
    rows.sort(key=lambda row: row["closed"] or "", reverse=True)
    json.dump(rows, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
