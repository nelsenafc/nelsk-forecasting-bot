"""The hourly data job: refresh the bot's resolved history and shadow-test the agent candidate.

Runs in the bot-data workflow with the repo's bot-data branch checked out in BOT_DATA_DIR. It
writes history.json [for the similar-questions lookup] and shadow.jsonl there, and the workflow
commits them. Prints counts only.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from forecasting_tools import MetaculusClient

import history
import run
import scorecard
import shadow
from forecast_log import bot_comment_index

HISTORY_MAX_AGE = timedelta(hours=6)
MIN_CREDIT_FOR_SHADOW = 10.0  # leave the rest for the live bot


def refresh_history(data_dir: Path, now: datetime) -> int | None:
    """Rebuild history.json when it is older than HISTORY_MAX_AGE. Returns the question count, or None if fresh."""
    path = data_dir / "history.json"
    if path.exists():
        updated = datetime.fromisoformat(json.loads(path.read_text())["updated_at"])
        if now - updated < HISTORY_MAX_AGE:
            return None
    cache_path = data_dir / "scorecard_cache.json"
    cache = scorecard.load_cache(cache_path)
    client = MetaculusClient()
    headers = {"Authorization": f"Token {os.environ['METACULUS_TOKEN']}"}
    comments = bot_comment_index(headers, client.get_current_user_id())
    records, _ = scorecard.collect(client, headers, comments, cache)
    scorecard.save_cache(cache_path, cache)
    path.write_text(history.dumps(history.build(records)))
    return len(records)


def main() -> None:
    data_dir = Path(os.getenv("BOT_DATA_DIR", "data"))
    data_dir.mkdir(exist_ok=True)
    count = refresh_history(data_dir, datetime.now(timezone.utc))
    print("History: fresh, not rebuilt" if count is None else f"History: {count} resolved questions")
    credit = run.openrouter_credit_left()
    if credit is not None and credit < MIN_CREDIT_FOR_SHADOW:
        print(f"Shadow: skipped, ${credit:.2f} of credit left is kept for the live bot")
        return
    written = asyncio.run(shadow.run(data_dir))
    print(f"Shadow: {written} new agent forecast(s)")


if __name__ == "__main__":
    main()
