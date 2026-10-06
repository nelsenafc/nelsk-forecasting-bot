from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from forecasting_tools import BinaryQuestion
from forecasting_tools.data_models.questions import QuestionState
from forecasting_tools.data_models.timestamped_predictions import BinaryTimestampedPrediction

import notion_sync

NOW = datetime(2026, 10, 6, 5, 37, tzinfo=timezone.utc)
HOUR = int(NOW.timestamp() // 3600)
COMMENT = """### How the final forecast was made
Final forecast: 6%
- openai/gpt-6.1-sol: 6%
- anthropic/claude-sonnet-5.5: 7%
"""


def question(post_id, state, resolution=None, score=None):
    return BinaryQuestion(
        question_text=f"Question {post_id}?",
        page_url=f"https://www.metaculus.com/questions/{post_id}/",
        state=state,
        tournament_slugs=["minibench-2026-10-05"],
        close_time=datetime(2026, 10, 5, 18, 39, tzinfo=timezone.utc),
        scheduled_resolution_time=datetime(2026, 10, 15, tzinfo=timezone.utc),
        api_json={
            "question": {
                "my_forecasts": {"latest": {}, "score_data": {} if score is None else {"spot_peer_score": score}},
                "resolution": resolution,
            }
        },
        previous_forecasts=[BinaryTimestampedPrediction(prediction_in_decimal=0.06, timestamp=NOW, timestamp_end=None)],
    )


def logged(post_id, status, **values):
    """A row as an earlier sync wrote it."""
    row = {
        "question": f"Question {post_id}?",
        "url": f"https://www.metaculus.com/questions/{post_id}/",
        "tournament": "MiniBench",
        "type": "Binary",
        "status": status,
        "closed": "2026-10-05",
        "resolves": "2026-10-15",
        "forecast": "",
        "models": "",
        "outcome": "",
        "peer_score": None,
    }
    return row | values


def as_read(properties):
    """Properties in the shape Notion returns them, given the shape notion_sync writes."""
    read = {}
    for name, value in properties.items():
        [(kind, content)] = value.items()
        if kind in ("title", "rich_text"):
            content = [{"plain_text": part["text"]["content"]} for part in content]
        read[name] = {"type": kind, kind: content}
    return read


class Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeNotion:
    def __init__(self, rows):
        self.pages = [
            {"id": f"page-{post_id}", "properties": as_read(notion_sync.row_properties(row))}
            for post_id, row in rows.items()
        ]
        self.writes = []
        self.headline = None
        self.last_edited = None

    def post(self, url, json, timeout):
        if url.endswith("/query"):
            return Response({"results": self.pages, "has_more": False})
        self.writes.append(("create", json["properties"]))
        return Response({"id": "new", "properties": as_read(json["properties"])})

    def patch(self, url, json, timeout):
        if "/blocks/" in url:
            self.headline = "".join(part["text"]["content"] for part in json["paragraph"]["rich_text"])
        elif url.endswith(notion_sync.LOG_PAGE_ID):
            self.last_edited = json["properties"]["Last edited"]["date"]["start"]
        else:
            self.writes.append(("update", json["properties"]))
            return Response({"id": url.rsplit("/", 1)[-1], "properties": as_read(json["properties"])})
        return Response({})

    def get(self, url, params, timeout):
        block = {"id": "b1", "type": "paragraph", "paragraph": {"rich_text": [{"plain_text": "10 forecasts logged, 0 resolved."}]}}
        return Response({"results": [block]})


@pytest.fixture
def sync(monkeypatch):
    def run(rows, questions):
        result = SimpleNamespace(notion=FakeNotion(rows), fetched=[], texts_read=[], exit_code=0)

        class Client:
            def get_current_user_id(self):
                return 7

            def get_question_by_post_id(self, post_id):
                result.fetched.append(post_id)
                return questions[post_id]

        def read_text(headers, comment):
            result.texts_read.append(comment["on_post"])
            return COMMENT

        posts = sorted(set(rows) | set(questions))
        monkeypatch.setenv("METACULUS_TOKEN", "test")
        monkeypatch.setattr(notion_sync, "notion_session", lambda: result.notion)
        monkeypatch.setattr(notion_sync, "MetaculusClient", Client)
        monkeypatch.setattr(notion_sync, "bot_comment_index", lambda headers, bot_id: {p: {"on_post": p} for p in posts})
        monkeypatch.setattr(notion_sync, "comment_text", read_text)
        monkeypatch.setattr(notion_sync, "WRITE_PAUSE_SECONDS", 0)
        try:
            notion_sync.main(now=NOW)
        except SystemExit as exit:
            result.exit_code = exit.code
        return result

    return run


def test_a_new_open_question_gets_a_row_without_numbers(sync):
    result = sync({}, {1: question(1, QuestionState.OPEN)})
    [(kind, properties)] = result.notion.writes
    assert kind == "create"
    assert properties["Status"]["select"]["name"] == "Open"
    assert properties["Bot forecast"]["rich_text"] == [] and properties["Models"]["rich_text"] == []
    assert result.texts_read == []
    assert result.notion.headline.startswith("1 forecast logged, 0 resolved.")
    assert result.notion.last_edited == "2026-10-06"


def test_closing_fills_in_the_numbers(sync):
    result = sync({2: logged(2, "Open")}, {2: question(2, QuestionState.CLOSED)})
    [(kind, properties)] = result.notion.writes
    assert kind == "update"
    assert properties["Bot forecast"]["rich_text"][0]["text"]["content"] == "6%"
    assert properties["Models"]["rich_text"][0]["text"]["content"] == "gpt-6.1-sol 6%; claude-sonnet-5.5 7%"
    assert result.texts_read == [2]


def test_resolution_adds_the_outcome_and_score_without_rereading_the_comment(sync):
    rows = {3: logged(3, "Closed", resolves="2026-10-06", forecast="6%", models="gpt-6.1-sol 6%")}
    result = sync(rows, {3: question(3, QuestionState.RESOLVED, resolution="no", score=3.14)})
    [(kind, properties)] = result.notion.writes
    assert properties["Outcome"]["rich_text"][0]["text"]["content"] == "no"
    assert properties["Peer score"]["number"] == 3.1
    assert properties["Models"]["rich_text"][0]["text"]["content"] == "gpt-6.1-sol 6%"
    assert result.texts_read == []
    assert result.notion.headline == "1 forecast logged, 1 resolved. Average peer score so far: +3.1 [above 0 beats the average bot]."


def test_settled_rows_are_not_fetched_again(sync):
    rows = {
        4: logged(4, "Resolved", forecast="6%", outcome="yes", peer_score=4.2),
        5: logged(5, "Resolved", forecast="6%", outcome="annulled"),
    }
    result = sync(rows, {})
    assert result.fetched == [] and result.notion.writes == []
    assert result.notion.headline is None and result.notion.last_edited is None


def test_closed_rows_are_checked_daily_until_due_then_hourly(sync):
    todays_turn = 24 - HOUR % 24  # (post id + hour) % 24 == 0
    settled = {"forecast": "6%", "models": "gpt-6.1-sol 6%"}
    rows = {
        todays_turn: logged(todays_turn, "Closed", **settled),
        todays_turn + 1: logged(todays_turn + 1, "Closed", **settled),
        todays_turn + 2: logged(todays_turn + 2, "Closed", resolves="2026-10-06", **settled),
    }
    questions = {post: question(post, QuestionState.CLOSED) for post in (todays_turn, todays_turn + 2)}
    result = sync(rows, questions)
    assert result.fetched == [todays_turn, todays_turn + 2]
    assert result.notion.writes == [] and result.texts_read == []


def test_a_question_that_fails_to_load_does_not_block_the_rest(sync):
    result = sync({9: logged(9, "Open")}, {8: question(8, QuestionState.OPEN)})
    assert [kind for kind, _ in result.notion.writes] == ["create"]
    assert result.exit_code == 1
