"""Offline end-to-end run: real bot plumbing, fake LLMs and fake markets.

Checks that research, the multi-model ensemble, parsing and combining fit
together for each question type without spending a cent.
"""

import asyncio
import re
import typing
from datetime import datetime, timezone

import pytest
from forecasting_tools import (
    BinaryPrediction,
    BinaryQuestion,
    GeneralLlm,
    MultipleChoiceQuestion,
    NumericQuestion,
    Percentile,
    PredictedOptionList,
)
from forecasting_tools.data_models.multiple_choice_report import PredictedOption

import forecaster
import markets
from ensemble import ForecasterSpec, combine_binary
from forecaster import EnsembleBot, MarketMatch

BINARY_ANSWERS = {"openai": 30, "anthropic": 40, "google": 20}

FORECASTERS = [
    ForecasterSpec(lab="openai", model="openrouter/openai/fake-a", samples=2),
    ForecasterSpec(lab="anthropic", model="openrouter/anthropic/fake-b", samples=1),
    ForecasterSpec(lab="google", model="openrouter/google/fake-c", samples=1),
]


def _lab_of(model: str) -> str:
    for lab in BINARY_ANSWERS:
        if f"/{lab}/" in model:
            return lab
    return "other"


async def fake_invoke(self, prompt, *args, **kwargs):
    text = prompt if isinstance(prompt, str) else str(prompt)
    lab = _lab_of(self.model)
    if "research assistant for a superforecaster" in text:
        return "Bank Indonesia held its rate in October 2026. Inflation is 2.1%."
    if "Write 2 short search queries" in text:
        return "Bank Indonesia rate cut November\nBI rate decision 2026"
    if "checking which prediction markets" in text:
        return "[0] exact - same meeting and criteria."
    if "resolves YES" in text:
        return f"Base rate reasoning...\nProbability: {BINARY_ANSWERS[lab]}%"
    if "probability of each option" in text:
        return "Hold: 60%\nCut: 30%\nHike: 10%" if lab == "openai" else "Hold: 50%\nCut: 40%\nHike: 10%"
    if "distribution for this quantity" in text:
        shift = {"openai": 0.0, "anthropic": 0.2, "google": -0.2}[lab]
        return "\n".join(
            f"Percentile {p}: {2.0 + shift + (p - 50) / 50:.2f}" for p in (5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95)
        )
    return "Summary of research."


async def fake_structure_output(text_to_structure, output_type, *args, **kwargs):
    text = text_to_structure
    if output_type is BinaryPrediction:
        return BinaryPrediction(prediction_in_decimal=int(re.findall(r"Probability: (\d+)%", text)[-1]) / 100)
    if output_type is PredictedOptionList:
        options = re.findall(r"^(\w+): (\d+)%", text, flags=re.M)
        return PredictedOptionList(
            predicted_options=[PredictedOption(option_name=name, probability=int(p) / 100) for name, p in options]
        )
    if typing.get_args(output_type) == (Percentile,):
        pairs = re.findall(r"Percentile (\d+): ([-\d.]+)", text)
        return [Percentile(percentile=int(p) / 100, value=float(v)) for p, v in pairs]
    if typing.get_args(output_type) == (MarketMatch,):
        return [MarketMatch(index=0, match="exact", note="Same meeting and criteria.")]
    raise AssertionError(f"Unexpected output type {output_type}")


QUOTE = markets.MarketQuote(
    source="Polymarket",
    title="Will Bank Indonesia cut rates in November 2026?",
    url="https://polymarket.com/event/bi-november",
    outcomes=[("Yes", 0.34), ("No", 0.66)],
    volume=125000,
)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(GeneralLlm, "invoke", fake_invoke)
    monkeypatch.setattr(forecaster, "structure_output", fake_structure_output)
    monkeypatch.setattr(markets, "search_all", lambda queries, per_source_limit=5: [QUOTE])
    for name in ("ASKNEWS_API_KEY", "ASKNEWS_CLIENT_ID", "ASKNEWS_SECRET"):
        monkeypatch.delenv(name, raising=False)


def make_bot() -> EnsembleBot:
    return EnsembleBot(
        forecasters=FORECASTERS,
        publish_reports_to_metaculus=False,
        skip_previously_forecasted_questions=False,
    )


COMMON = dict(
    background_info="BI policy rate is 4.75%.",
    resolution_criteria="Resolves per the BI press release after the November meeting.",
    fine_print="Unscheduled meetings do not count.",
    close_time=datetime(2026, 10, 10, tzinfo=timezone.utc),
    scheduled_resolution_time=datetime(2026, 11, 20, tzinfo=timezone.utc),
)


def test_binary_question_end_to_end():
    question = BinaryQuestion(
        question_text="Will Bank Indonesia cut its policy rate in November 2026?",
        id_of_post=1,
        page_url="https://www.metaculus.com/questions/1/",
        **COMMON,
    )
    report = asyncio.run(make_bot().forecast_question(question))
    expected = combine_binary([0.30, 0.30, 0.40, 0.20], ["openai", "openai", "anthropic", "google"])
    assert report.prediction == pytest.approx(expected)
    assert "How the final forecast was made" in report.explanation
    assert "Prediction markets" in report.explanation


def test_multiple_choice_question_end_to_end():
    question = MultipleChoiceQuestion(
        question_text="What will Bank Indonesia do in November 2026?",
        id_of_post=2,
        page_url="https://www.metaculus.com/questions/2/",
        options=["Hold", "Cut", "Hike"],
        **COMMON,
    )
    report = asyncio.run(make_bot().forecast_question(question))
    probabilities = {option.option_name: option.probability for option in report.prediction.predicted_options}
    assert list(probabilities) == ["Hold", "Cut", "Hike"]
    assert sum(probabilities.values()) == pytest.approx(1.0)
    # Lab-weighted: openai 60/30/10, anthropic and google 50/40/10.
    assert probabilities["Hold"] == pytest.approx((0.6 + 0.5 + 0.5) / 3, abs=0.01)


def test_numeric_question_end_to_end():
    question = NumericQuestion(
        question_text="What will Indonesia's year-on-year CPI inflation be for October 2026?",
        id_of_post=3,
        page_url="https://www.metaculus.com/questions/3/",
        upper_bound=6.0,
        lower_bound=-2.0,
        open_upper_bound=True,
        open_lower_bound=True,
        unit_of_measure="%",
        **COMMON,
    )
    report = asyncio.run(make_bot().forecast_question(question))
    cdf = report.prediction.get_cdf()
    heights = [point.percentile for point in cdf]
    assert all(b >= a for a, b in zip(heights, heights[1:]))
    median = next(point.value for point in cdf if point.percentile >= 0.5)
    assert 1.6 < median < 2.4


def test_bot_refuses_to_forecast_blind(monkeypatch):
    async def failing_web_research(self, question):
        raise RuntimeError("search down")

    monkeypatch.setattr(EnsembleBot, "_web_research", failing_web_research)
    question = BinaryQuestion(
        question_text="Will it rain?", id_of_post=4, page_url="https://www.metaculus.com/questions/4/", **COMMON
    )
    with pytest.raises(Exception):
        asyncio.run(make_bot().forecast_question(question))


def test_forecast_survives_one_lab_failing(monkeypatch):
    async def flaky_invoke(self, prompt, *args, **kwargs):
        if "/google/" in self.model and "resolves YES" in str(prompt):
            raise TimeoutError("google timed out")
        return await fake_invoke(self, prompt, *args, **kwargs)

    monkeypatch.setattr(GeneralLlm, "invoke", flaky_invoke)
    question = BinaryQuestion(
        question_text="Will Bank Indonesia cut?", id_of_post=5, page_url="https://www.metaculus.com/questions/5/", **COMMON
    )
    report = asyncio.run(make_bot().forecast_question(question))
    expected = combine_binary([0.30, 0.30, 0.40], ["openai", "openai", "anthropic"])
    assert report.prediction == pytest.approx(expected)
    assert "failed and were left out" in report.explanation
