import history

ENTRIES = [
    {
        "post_id": 1,
        "question": "Will anyone besides Gianni Infantino have publicly declared a candidacy for the 2027 FIFA presidency?",
        "type": "Binary",
        "resolution": "yes",
        "our_forecast": "9%",
        "peer_score": -59.9,
        "resolved_on": "2026-10-16",
    },
    {
        "post_id": 2,
        "question": "Will Delta Air Lines (DAL) stock close above $85.00 on October 15, 2026?",
        "type": "Binary",
        "resolution": "no",
        "our_forecast": "36%",
        "peer_score": 8.6,
        "resolved_on": "2026-10-15",
    },
]


def test_similar_finds_the_matching_template_and_skips_the_question_itself():
    found = history.similar("Will anyone declare a candidacy for FIFA president before the 2027 congress?", ENTRIES)
    assert [item["post_id"] for item in found] == [1]
    assert history.similar(ENTRIES[0]["question"], ENTRIES, exclude=1) == []
    assert history.similar("the of will by", ENTRIES) == []


def test_format_similar_shows_outcome_forecast_and_score():
    text = history.format_similar(ENTRIES[:1])
    assert "resolved yes; the bot forecast 9%; peer score -60" in text
    assert "not as answers" in text
    assert history.format_similar([]) == ""


def test_describe_each_forecast_shape():
    assert history.describe({"final": 0.095}) == "10%"
    assert history.describe({"final": {"Arsenal": 0.7, "Chelsea": 0.3}}) == "Arsenal 70% / Chelsea 30%"
    assert history.describe({"final": [2.0, 2.8, 4.1]}) == "median 2.8 [80% range 2 to 4.1]"
    assert history.describe({"final": None}) == "not recorded"


def test_load_survives_an_unreachable_branch(monkeypatch):
    def fail(*args, **kwargs):
        raise history.requests.ConnectionError("offline")

    monkeypatch.setattr(history, "_loaded", None)
    monkeypatch.setattr(history.requests, "get", fail)
    assert history.load() == []
