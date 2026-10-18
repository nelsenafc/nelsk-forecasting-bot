"""Run the agent candidate on tournament questions right after they close, and keep its forecasts.

The agent forecasts each question the live bot forecast once that question has closed, so no
one sees a forecast on an open question and the live bot's entry stays the only one submitted.
Nothing is published. Questions are taken only within hours of closing, so the agent sees about
what the live bot saw. It does get up to about 3 extra hours of news, which the bar for
switching [+3 points a question over 40 questions] allows for. The scorecard pairs each shadow
forecast with the live one once the question resolves.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from forecasting_tools import MetaculusClient
from forecasting_tools.data_models.questions import QuestionState

from agent import AGENT_MODEL, AgentBot
from forecast_log import bot_comment_index, tournament_name, type_name
from forecaster import compact_prediction
from guards import may_preview

SHADOW_FILE = "shadow.jsonl"
SHADOW_URL = "https://raw.githubusercontent.com/nelsenafc/nelsk-forecasting-bot/bot-data/shadow.jsonl"
CLOSED_WITHIN = timedelta(hours=int(os.getenv("SHADOW_CLOSED_WITHIN_HOURS", "6")))
MAX_PER_RUN = int(os.getenv("SHADOW_MAX_PER_RUN", "6"))


def read_rows(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_rows(path: Path | None = None) -> list[dict]:
    """Shadow forecasts from a local data checkout, or from the bot-data branch."""
    if path is not None:
        return read_rows(path.read_text()) if path.exists() else []
    try:
        response = requests.get(SHADOW_URL, timeout=15)
        return read_rows(response.text) if response.ok else []
    except requests.RequestException:
        return []


def due(client, headers: dict, done: set[int], now: datetime) -> list:
    """Tournament questions the live bot forecast that closed within the window and have no shadow yet."""
    # The bot comments while a question is open, at most 3 hours before it closes, so older comments are skipped unread.
    earliest = (now - CLOSED_WITHIN - timedelta(hours=4)).isoformat()
    found = []
    for post_id, comment in bot_comment_index(headers, client.get_current_user_id()).items():
        if post_id in done or (comment.get("created_at") or "") < earliest:
            continue
        question = client.get_question_by_post_id(post_id)
        closed = question.close_time
        if (
            tournament_name(question)
            and question.state == QuestionState.CLOSED
            and closed is not None
            and now - CLOSED_WITHIN <= closed <= now
        ):
            found.append(question)
    return found


async def run(data_dir: Path, now: datetime | None = None, bot: AgentBot | None = None) -> int:
    """Forecast due questions with the agent and append them to shadow.jsonl. Returns how many were written."""
    now = now or datetime.now(timezone.utc)
    path = data_dir / SHADOW_FILE
    done = {row["post_id"] for row in load_rows(path)}
    client = MetaculusClient()
    headers = {"Authorization": f"Token {os.environ['METACULUS_TOKEN']}"}
    questions = due(client, headers, done, now)[:MAX_PER_RUN]
    bot = bot or AgentBot(publish_reports_to_metaculus=False, skip_previously_forecasted_questions=False)

    async def forecast(question):
        try:
            return question, await bot.forecast_question(question)
        except Exception as error:
            print(f"Shadow forecast failed for post {question.id_of_post}: {type(error).__name__}")
            return question, None

    # Several at once: an agent run takes 3-6 minutes, and the bot's own limiter caps research at two.
    # may_preview keeps out any tournament question that is still open.
    results = await asyncio.gather(*(forecast(q) for q in questions if may_preview(q)))
    written = 0
    for question, report in results:
        if report is None:
            continue
        agent = bot.agent_runs.get(question.page_url or question.question_text)
        row = {
            "post_id": question.id_of_post,
            "type": type_name(question),
            "closed_at": question.close_time.isoformat(),
            "made_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": AGENT_MODEL,
            "forecast": compact_prediction(report.prediction),
            "tool_calls": agent.tool_calls if agent else None,
            "research_cost": round(agent.cost, 4) if agent else None,
        }
        with path.open("a") as file:
            file.write(json.dumps(row) + "\n")
        written += 1
    return written
