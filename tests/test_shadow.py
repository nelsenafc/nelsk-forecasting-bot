import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from forecasting_tools import BinaryQuestion
from forecasting_tools.data_models.questions import QuestionState

import shadow

NOW = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)


def question(post_id, state, closed_hours_ago):
    return BinaryQuestion(
        question_text=f"Question {post_id}?",
        id_of_post=post_id,
        page_url=f"https://www.metaculus.com/questions/{post_id}/",
        state=state,
        tournament_slugs=["minibench-2026-10-19"],
        close_time=NOW - timedelta(hours=closed_hours_ago),
    )


QUESTIONS = {
    1: question(1, QuestionState.CLOSED, 1),  # due
    2: question(2, QuestionState.OPEN, -1),  # still open: never
    3: question(3, QuestionState.CLOSED, 10),  # closed too long ago
    4: question(4, QuestionState.CLOSED, 2),  # already shadowed
}
COMMENTS = {post_id: {"created_at": (NOW - timedelta(hours=3)).isoformat()} for post_id in QUESTIONS}


class FakeClient:
    def get_current_user_id(self):
        return 7

    def get_question_by_post_id(self, post_id):
        return QUESTIONS[post_id]


class FakeBot:
    def __init__(self):
        self.forecasted = []
        self.agent_runs = {}

    async def forecast_question(self, q):
        self.forecasted.append(q.id_of_post)
        self.agent_runs[q.page_url] = SimpleNamespace(tool_calls=5, cost=0.0612)
        return SimpleNamespace(prediction=0.3)


def test_due_takes_only_recently_closed_unshadowed_questions(monkeypatch):
    monkeypatch.setattr(shadow, "bot_comment_index", lambda headers, bot_id: COMMENTS)
    found = shadow.due(FakeClient(), {}, done={4}, now=NOW)
    assert [q.id_of_post for q in found] == [1]


def test_run_appends_one_row_per_shadow_forecast(monkeypatch, tmp_path):
    (tmp_path / shadow.SHADOW_FILE).write_text(json.dumps({"post_id": 4, "forecast": "20%"}) + "\n")
    monkeypatch.setattr(shadow, "bot_comment_index", lambda headers, bot_id: COMMENTS)
    monkeypatch.setattr(shadow, "MetaculusClient", FakeClient)
    monkeypatch.setenv("METACULUS_TOKEN", "test")
    bot = FakeBot()
    assert asyncio.run(shadow.run(tmp_path, now=NOW, bot=bot)) == 1
    assert bot.forecasted == [1]
    rows = shadow.load_rows(tmp_path / shadow.SHADOW_FILE)
    assert rows[-1]["post_id"] == 1 and rows[-1]["forecast"] == "30%"
    assert rows[-1]["tool_calls"] == 5 and rows[-1]["research_cost"] == 0.0612


def test_run_stops_at_the_daily_cap(monkeypatch, tmp_path):
    recent = (NOW - timedelta(hours=1)).isoformat()
    lines = [json.dumps({"post_id": 100 + i, "forecast": "50%", "made_at": recent}) for i in range(shadow.MAX_PER_DAY)]
    (tmp_path / shadow.SHADOW_FILE).write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(shadow, "bot_comment_index", lambda headers, bot_id: COMMENTS)
    monkeypatch.setattr(shadow, "MetaculusClient", FakeClient)
    monkeypatch.setenv("METACULUS_TOKEN", "test")
    bot = FakeBot()
    assert asyncio.run(shadow.run(tmp_path, now=NOW, bot=bot)) == 0
    assert bot.forecasted == []
