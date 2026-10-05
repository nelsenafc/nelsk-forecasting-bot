from datetime import datetime, timezone

from forecasting_tools import BinaryQuestion
from forecasting_tools.data_models.questions import QuestionState

from forecast_log import build_row, model_summary

COMMENT = """# SUMMARY
## R1: Forecaster 1 Reasoning
### How the final forecast was made
Final forecast: 6%
Each model forecast independently from the same research.
- openai/gpt-6.1-sol: 6%
- anthropic/claude-sonnet-5.5: 7%
- google/gemini-3.8-flash: median 101.2 [80% range 89.5 to 115]
- openai/gpt-6.1-sol: Probability distribution:
### Forecaster 1: openai/gpt-6.1-sol [6%]
Reasoning text.
"""


def test_model_summary_reads_the_compact_lines():
    assert model_summary(COMMENT) == (
        "gpt-6.1-sol 6%; claude-sonnet-5.5 7%; gemini-3.8-flash median 101.2 [80% range 89.5 to 115]"
    )
    assert model_summary("no summary here") == ""


def make(state):
    return BinaryQuestion(
        question_text="Will the US strike Iran on 15-16 Oct?",
        page_url="https://www.metaculus.com/questions/1/",
        state=state,
        tournament_slugs=["minibench-2026-10-05"],
        close_time=datetime(2026, 10, 5, 18, 39, tzinfo=timezone.utc),
        api_json={"question": {"my_forecasts": {"latest": {}, "score_data": {}}, "resolution": None}},
        previous_forecasts=[],
    )


def test_open_rows_never_show_the_forecast():
    row = build_row(make(QuestionState.OPEN), COMMENT, "MiniBench")
    assert row["status"] == "Open"
    assert row["forecast"] == "" and row["models"] == ""


def test_closed_rows_show_the_model_lines():
    row = build_row(make(QuestionState.CLOSED), COMMENT, "MiniBench")
    assert row["status"] == "Closed"
    assert row["models"].startswith("gpt-6.1-sol 6%")
