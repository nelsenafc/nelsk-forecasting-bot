import math
from datetime import datetime, timezone

import pytest
from forecasting_tools import BinaryQuestion, MultipleChoiceQuestion, NumericQuestion
from forecasting_tools.data_models.questions import QuestionState

from scorecard import build_record, combine, comment_forecasts, log_score, parse_forecast, range_loss, report

SLUGS = ["minibench-2026-10-05"]
RESOLVED = datetime(2026, 10, 16, tzinfo=timezone.utc)


def comment(*lines):
    return "# SUMMARY\n### How the final forecast was made\n" + "\n".join(lines) + "\n### Forecaster 1: x [y]\nReasoning.\n"


def api(resolution, peer_score=None):
    scores = {} if peer_score is None else {"spot_peer_score": peer_score}
    return {"question": {"resolution": resolution, "my_forecasts": {"score_data": scores}}}


def binary(post_id, resolution, final, labs, peer_score):
    question = BinaryQuestion(
        question_text=f"Binary {post_id}?",
        id_of_post=post_id,
        page_url=f"https://www.metaculus.com/questions/{post_id}/",
        state=QuestionState.RESOLVED,
        tournament_slugs=SLUGS,
        actual_resolution_time=RESOLVED,
        api_json=api(resolution, peer_score),
    )
    lines = [f"Final forecast: {final}"] + [f"- {model}: {value}" for model, value in labs]
    return build_record(question, comment(*lines), "MiniBench")


def test_parse_forecast_reads_every_format():
    assert parse_forecast("6%", "Binary") == 0.06
    assert parse_forecast("9.5%", "Binary") == 0.095
    assert parse_forecast("median 1,234 [80% range 1,000 to 1,500]", "Numeric") == [1000, 1234, 1500]
    assert parse_forecast("median -21.7 [80% range -30.2 to -12]", "Numeric") == [-30.2, -21.7, -12]
    assert parse_forecast("median 1.23e-05 [80% range 1e-05 to 2e-05]", "Count") == [1e-05, 1.23e-05, 2e-05]
    low, median, high = parse_forecast("median 2026-10-15 [80% range 2026-10-10 to 2026-10-20]", "Date")
    assert median == datetime(2026, 10, 15, tzinfo=timezone.utc).timestamp() and low < median < high
    options = ["Yes, before 2027", "No / not yet"]
    assert parse_forecast("Yes, before 2027 60% / No / not yet 40%", "Multiple choice", options) == {
        "Yes, before 2027": 0.6,
        "No / not yet": 0.4,
    }
    assert parse_forecast("Probability distribution:", "Numeric") is None


def test_comment_forecasts_groups_labs_and_reads_the_price_line():
    text = comment(
        "Final forecast: median 101 [80% range 92 to 110]",
        "- openai/gpt-6.1-sol: median 100 [80% range 90 to 112]",
        "- openai/gpt-6.1-sol: median 102 [80% range 91 to 114]",
        "- anthropic/claude-sonnet-5.5: median 99 [80% range 88 to 111]",
        "- stats/price-model: median 101 [80% range 94 to 108] [yahoo BZ=F, 50% of the final forecast; "
        "AI ensemble alone: median 100 [80% range 89 to 112]]",
    )
    found = comment_forecasts(text, "Numeric")
    assert found["final"] == [92, 101, 110]
    assert {lab: len(samples) for lab, samples in found["labs"].items()} == {"openai": 2, "anthropic": 1}
    assert found["price_model"] == [94, 101, 108]
    assert found["ai_alone"] == [89, 100, 112]


def test_combine_matches_the_ensemble():
    assert combine([0.1, 0.9], "Binary") == pytest.approx(0.5)
    assert combine([0.2, 0.2], "Binary") == pytest.approx(0.2)
    assert combine([[1, 2, 3], [3, 4, 5]], "Numeric") == [2, 3, 4]


def test_scores():
    assert log_score(0.8, {"type": "Binary", "resolution": "yes"}) == pytest.approx(100 * math.log(0.8))
    assert log_score(0.8, {"type": "Binary", "resolution": "no"}) == pytest.approx(100 * math.log(0.2))
    record = {"type": "Numeric", "resolution": 20.0, "lower": 0.0, "upper": 100.0}
    assert range_loss([10, 20, 30], record) == pytest.approx(2 / 3)
    assert range_loss([20, 20, 20], record) == 0


def test_unresolved_and_annulled_questions_are_not_scored():
    assert binary(1, None, "6%", [], None) is None
    assert binary(2, "annulled", "6%", [], None) is None


def test_numeric_and_multiple_choice_records():
    numeric = NumericQuestion(
        question_text="Brent on 16 Oct?",
        id_of_post=3,
        state=QuestionState.RESOLVED,
        tournament_slugs=SLUGS,
        upper_bound=150.0,
        lower_bound=50.0,
        open_upper_bound=True,
        open_lower_bound=True,
        api_json=api("above_upper_bound", -3.2),
    )
    record = build_record(numeric, comment("Final forecast: median 101 [80% range 92 to 110]"), "MiniBench")
    assert record["resolution"] == 150.0 and record["final"] == [92, 101, 110]

    choice = MultipleChoiceQuestion(
        question_text="Who wins?",
        id_of_post=4,
        state=QuestionState.RESOLVED,
        tournament_slugs=SLUGS,
        options=["Arsenal", "Chelsea"],
        api_json=api("Arsenal", 8.0),
    )
    text = comment("Final forecast: Arsenal 70% / Chelsea 30%", "- google/gemini-3.8-flash: Arsenal 65% / Chelsea 35%")
    record = build_record(choice, text, "MiniBench")
    assert record["labs"]["google"] == {"Arsenal": 0.65, "Chelsea": 0.35}
    assert build_record(choice.model_copy(update={"api_json": api("Spurs")}), text, "MiniBench") is None


def test_report_covers_the_field_calibration_and_labs():
    labs_right = [("openai/gpt-6.1-sol", "80%"), ("anthropic/claude-sonnet-5.5", "70%"), ("google/gemini-3.8-flash", "40%")]
    labs_wrong = [("openai/gpt-6.1-sol", "10%"), ("anthropic/claude-sonnet-5.5", "20%"), ("google/gemini-3.8-flash", "60%")]
    records = [binary(10, "yes", "70%", labs_right, 12.0), binary(11, "no", "20%", labs_wrong, 6.0)]
    text = report(records, failed=0, today="2026-10-16")
    assert "2 resolved question(s): 2 binary." in text
    assert "| MiniBench | 2 | +18.0 | +9.0 |" in text
    assert "Brier score" in text
    google = next(line for line in text.splitlines() if line.startswith("| google |"))
    without = float(google.split("|")[5])
    assert without > 0  # google pulled both forecasts the wrong way, so the rest do better without it
    assert "## Price model" not in text


def test_report_with_nothing_resolved():
    assert "No resolved tournament questions yet." in report([], failed=0, today="2026-10-06")
