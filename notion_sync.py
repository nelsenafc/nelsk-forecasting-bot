"""Keep the Notion forecast log in step with the bot's Metaculus forecasts.

Runs hourly on GitHub. Rows are matched by Metaculus post id:
- a new forecast gets a row with the question only,
- its numbers fill in once the question closes [forecast_log.py enforces that],
- the outcome and the bot's peer score fill in once it resolves.
Each run re-reads only the questions that can have changed: open ones, closed ones whose
scheduled resolution date has come, resolved ones still waiting for a score, and every
other closed one once a day, to catch early resolutions.
It prints counts only, because this repo's run logs are public.

Needs NOTION_TOKEN [an internal integration connected to the log page] and METACULUS_TOKEN.
"""

from __future__ import annotations

import os
import re
import time
from collections import Counter
from datetime import datetime, timezone

import requests
from forecasting_tools import MetaculusClient
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from forecast_log import bot_comment_index, build_row, comment_text, tournament_name
from guards import may_preview

NOTION_API = "https://api.notion.com/v1"
NOTION_VERSION = "2025-09-03"
DATA_SOURCE_ID = os.getenv("NOTION_FORECASTS_DATA_SOURCE_ID", "1c6bc105-6c43-4af2-aeff-58fee1280db8")
LOG_PAGE_ID = os.getenv("NOTION_LOG_PAGE_ID", "3f088ea99dbf81b58dcdeaf11ce26abf")
HEADLINE = re.compile(r"\d+ forecasts? logged")
POST_ID = re.compile(r"/questions/(\d+)")
UNSCORED_OUTCOMES = {"annulled", "ambiguous"}
TEXT_LIMIT = 1900  # Notion caps one text value at 2,000 characters
WRITE_PAUSE_SECONDS = 0.4  # Notion averages three requests a second


def notion_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {os.environ['NOTION_TOKEN']}",
            "Notion-Version": NOTION_VERSION,
            "Content-Type": "application/json",
        }
    )
    # A 429 means Notion wrote nothing, so retrying after its Retry-After wait is safe for any method.
    retry = Retry(total=5, status_forcelist=[429], allowed_methods=None, backoff_factor=1)
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _plain(rich_text: list[dict] | None) -> str:
    return "".join(part.get("plain_text", "") for part in rich_text or [])


def _status(page: dict) -> str | None:
    return (page["properties"]["Status"]["select"] or {}).get("name")


def row_properties(row: dict) -> dict:
    def text(value: str) -> dict:
        return {"rich_text": [{"text": {"content": value[:TEXT_LIMIT]}}] if value else []}

    def select(value: str | None) -> dict:
        return {"select": {"name": value} if value else None}

    def day(value: str | None) -> dict:
        return {"date": {"start": value} if value else None}

    return {
        "Question": {"title": [{"text": {"content": row["question"][:TEXT_LIMIT]}}]},
        "Tournament": select(row["tournament"]),
        "Type": select(row["type"]),
        "Status": select(row["status"]),
        "Closed": day(row["closed"]),
        "Resolves": day(row["resolves"]),
        "Bot forecast": text(row["forecast"]),
        "Models": text(row["models"]),
        "Outcome": text(row["outcome"]),
        "Peer score": {"number": row["peer_score"]},
        "Link": {"url": row["url"]},
    }


def page_signature(page: dict) -> tuple:
    props = page["properties"]
    return (
        _status(page),
        _plain(props["Bot forecast"]["rich_text"]),
        _plain(props["Models"]["rich_text"]),
        _plain(props["Outcome"]["rich_text"]),
        props["Peer score"]["number"],
    )


def row_signature(row: dict) -> tuple:
    return (
        row["status"],
        row["forecast"][:TEXT_LIMIT],
        row["models"][:TEXT_LIMIT],
        row["outcome"][:TEXT_LIMIT],
        row["peer_score"],
    )


def needs_check(page: dict, post_id: int, today: str, hour: int) -> bool:
    """Whether a logged question can have changed since its row was last written."""
    props = page["properties"]
    status = _status(page)
    if status == "Resolved":
        # final once the score is in; annulled and ambiguous questions never get one
        unscored = _plain(props["Outcome"]["rich_text"]).lower() in UNSCORED_OUTCOMES
        return props["Peer score"]["number"] is None and not unscored
    if status == "Closed":
        # hourly from the scheduled resolution date; before it, once a day per question,
        # spread across the hours
        resolves = (props["Resolves"]["date"] or {}).get("start") or ""
        return resolves[:10] <= today or (post_id + hour) % 24 == 0
    return True


def existing_pages(session: requests.Session) -> dict[int, dict]:
    """Logged rows by Metaculus post id."""
    pages: dict[int, dict] = {}
    body: dict = {"page_size": 100}
    while True:
        response = session.post(f"{NOTION_API}/data_sources/{DATA_SOURCE_ID}/query", json=body, timeout=30)
        response.raise_for_status()
        payload = response.json()
        for page in payload["results"]:
            match = POST_ID.search(page["properties"]["Link"]["url"] or "")
            if match:
                pages[int(match.group(1))] = page
        if not payload.get("has_more"):
            return pages
        body["start_cursor"] = payload["next_cursor"]


def update_headline(session: requests.Session, pages: list[dict]) -> None:
    """Rewrite the count line at the top of the log page."""
    resolved = [page for page in pages if _status(page) == "Resolved"]
    scores = [page["properties"]["Peer score"]["number"] for page in resolved]
    scores = [score for score in scores if score is not None]
    summary = f"{len(pages)} forecast{'' if len(pages) == 1 else 's'} logged, {len(resolved)} resolved."
    detail = (
        f" Average peer score so far: {sum(scores) / len(scores):+.1f} [above 0 beats the average bot]."
        if scores
        else " Scores appear here as questions resolve."
    )
    response = session.get(f"{NOTION_API}/blocks/{LOG_PAGE_ID}/children", params={"page_size": 20}, timeout=30)
    response.raise_for_status()
    for block in response.json()["results"]:
        kind = block["type"]
        rich_text = block.get(kind, {}).get("rich_text")
        if rich_text is not None and HEADLINE.search(_plain(rich_text)):
            text = [
                {"text": {"content": summary}, "annotations": {"bold": True}},
                {"text": {"content": detail}},
            ]
            session.patch(
                f"{NOTION_API}/blocks/{block['id']}", json={kind: {"rich_text": text}}, timeout=30
            ).raise_for_status()
            return


def main(now: datetime | None = None) -> None:
    now = now or datetime.now(timezone.utc)
    today, hour = now.date().isoformat(), int(now.timestamp() // 3600)
    session = notion_session()
    pages = existing_pages(session)
    client = MetaculusClient()
    headers = {"Authorization": f"Token {os.environ['METACULUS_TOKEN']}"}
    counts: Counter[str] = Counter()
    for post_id, comment in bot_comment_index(headers, client.get_current_user_id()).items():
        page = pages.get(post_id)
        if page and not needs_check(page, post_id, today, hour):
            counts["unchanged"] += 1
            continue
        try:
            question = client.get_question_by_post_id(post_id)
        except Exception as error:  # one unreadable question should not stall the whole log
            print(f"Could not read post {post_id}: {type(error).__name__}")
            counts["failed"] += 1
            continue
        tournament = tournament_name(question)
        if tournament is None:  # practice and main-site questions stay out of the log
            continue
        visible = may_preview(question)
        known_models = _plain(page["properties"]["Models"]["rich_text"]) if page else ""
        # An archived comment costs a rate-limited call, so its text is read once: when the
        # question has closed and its row has no per-model forecasts yet.
        text = comment_text(headers, comment) if visible and not known_models else ""
        row = build_row(question, text, tournament)
        if visible and known_models:
            row["models"] = known_models
        if page and page_signature(page) == row_signature(row):
            counts["unchanged"] += 1
            continue
        if page:
            response = session.patch(
                f"{NOTION_API}/pages/{page['id']}", json={"properties": row_properties(row)}, timeout=30
            )
        else:
            response = session.post(
                f"{NOTION_API}/pages",
                json={
                    "parent": {"type": "data_source_id", "data_source_id": DATA_SOURCE_ID},
                    "properties": row_properties(row),
                },
                timeout=30,
            )
        response.raise_for_status()
        counts["updated" if page else "new"] += 1
        pages[post_id] = response.json()
        time.sleep(WRITE_PAUSE_SECONDS)
    if counts["new"] or counts["updated"]:
        update_headline(session, list(pages.values()))
        session.patch(
            f"{NOTION_API}/pages/{LOG_PAGE_ID}",
            json={"properties": {"Last edited": {"date": {"start": today}}}},
            timeout=30,
        ).raise_for_status()
    print(
        f"Notion forecast log: {counts['new']} new, {counts['updated']} updated, "
        f"{counts['unchanged']} unchanged, {counts['failed']} failed"
    )
    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
