"""The forecasting bot: research once, forecast with several models, combine.

Built on the Metaculus Fall 2026 template [FallTemplateBot2026], which handles
question loading, conditional questions, comments and submission. This class
replaces three pieces:
- research: AskNews + a web-search model + matching prediction markets
- forecasting: every question goes to models from three labs
- combining: see ensemble.py
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
from datetime import datetime, timezone
from typing import Literal

from forecasting_tools import (
    AskNewsSearcher,
    BinaryPrediction,
    BinaryQuestion,
    ConditionalQuestion,
    DatePercentile,
    DateQuestion,
    GeneralLlm,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    PredictedOptionList,
    ReasonedPrediction,
    clean_indents,
    structure_output,
)
from forecasting_tools.data_models.multiple_choice_report import PredictedOption
from forecasting_tools.forecast_bots.official_bots.template_bot_2026_fall import (
    FallTemplateBot2026,
)
from pydantic import BaseModel

import markets
import prompts
from ensemble import (
    ForecasterSpec,
    combine_binary,
    combine_cdfs,
    combine_multiple_choice,
    default_forecasters,
)

logger = logging.getLogger(__name__)

CHEAP_MODEL = os.getenv("CHEAP_MODEL", "openrouter/openai/gpt-6-luna")
RESEARCH_MODEL = os.getenv("RESEARCH_MODEL", "openrouter/openai/gpt-6.1-sol")
# "native" keeps search on the model provider's own tool, which the Metaculus credits cover.
# Leaving the engine unset can fall back to Exa, which they don't.
RESEARCH_WEB_PLUGIN = {"id": "web", "engine": "native", "max_results": 8}
MAX_MARKET_CANDIDATES = 12


class MarketMatch(BaseModel):
    index: int
    match: Literal["exact", "close", "related"]
    note: str


def llm_for(spec: ForecasterSpec) -> GeneralLlm:
    extra: dict = {}
    if spec.reasoning_effort:
        # OpenRouter's unified reasoning setting; it maps to each lab's own thinking controls.
        extra["extra_body"] = {"reasoning": {"effort": spec.reasoning_effort}}
    return GeneralLlm(model=spec.model, timeout=spec.timeout, allowed_tries=2, **extra)


def _real_env(name: str) -> bool:
    value = (os.getenv(name) or "").strip()
    return bool(value) and value != "REPLACE_ME"


def has_asknews_credentials() -> bool:
    return _real_env("ASKNEWS_API_KEY") or (
        _real_env("ASKNEWS_CLIENT_ID") and _real_env("ASKNEWS_SECRET")
    )


def _pct(probability: float) -> str:
    return f"{probability * 100:.1f}".rstrip("0").rstrip(".") + "%"


def _number(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:,.1f}"
    return f"{value:.3g}"


def compact_prediction(value) -> str:
    """One-line summary of any prediction, for comments and the forecast log."""
    if isinstance(value, float):
        return _pct(value)
    if isinstance(value, PredictedOptionList):
        return " / ".join(
            f"{option.option_name} {_pct(option.probability)}" for option in value.predicted_options
        )
    if isinstance(value, NumericDistribution):
        p10, p50, p90 = (
            point.value for point in value.get_percentiles_at_target_heights([0.1, 0.5, 0.9])
        )
        if getattr(value, "is_date", False):
            show = lambda v: datetime.fromtimestamp(v, tz=timezone.utc).date().isoformat()
        else:
            show = _number
        return f"median {show(p50)} [80% range {show(p10)} to {show(p90)}]"
    return str(value)


def parse_queries(text: str, limit: int = 2) -> list[str]:
    queries = []
    for line in text.splitlines():
        query = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip("\"'")
        if query:
            queries.append(query)
    return queries[:limit]


class EnsembleBot(FallTemplateBot2026):
    _max_concurrent_questions = 2
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)
    _structure_output_validation_samples = 2
    min_labs = 2

    def __init__(
        self,
        *,
        forecasters: list[ForecasterSpec] | None = None,
        use_market_signals: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.forecasters = forecasters or default_forecasters()
        self.use_market_signals = use_market_signals

    @classmethod
    def _llm_config_defaults(cls) -> dict[str, str | GeneralLlm | None]:
        defaults = super()._llm_config_defaults()
        defaults.update(
            {
                # Only used if something falls back to the template's single-model path.
                "default": llm_for(default_forecasters()[0]),
                "researcher": GeneralLlm(
                    model=RESEARCH_MODEL,
                    timeout=300,
                    allowed_tries=2,
                    extra_body={
                        "plugins": [RESEARCH_WEB_PLUGIN],
                        "reasoning": {"effort": "medium"},
                    },
                ),
                "summarizer": GeneralLlm(model=CHEAP_MODEL, timeout=120, allowed_tries=2),
                "parser": GeneralLlm(model=CHEAP_MODEL, timeout=120, allowed_tries=3),
            }
        )
        return defaults

    ##################################### RESEARCH #####################################

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            sources = {
                "news": self._news_research(question),
                "web": self._web_research(question),
            }
            if self.use_market_signals:
                sources["markets"] = self._market_research(question)
            results = await asyncio.gather(*sources.values(), return_exceptions=True)

        sections: dict[str, str] = {}
        for name, result in zip(sources, results):
            if isinstance(result, BaseException):
                logger.warning(f"{name} research failed for {question.page_url}: {result}")
            elif result and result.strip():
                sections[name] = result.strip()
        if "news" not in sections and "web" not in sections:
            raise RuntimeError(
                f"Both news and web research failed for {question.page_url}; not forecasting blind"
            )
        return "\n\n".join(sections.values())

    async def _news_research(self, question: MetaculusQuestion) -> str:
        if not has_asknews_credentials():
            return ""
        news = await AskNewsSearcher().call_preconfigured_version(
            "asknews/news-summaries", question.question_text
        )
        return f"## News articles [AskNews]\n{news}"

    async def _web_research(self, question: MetaculusQuestion) -> str:
        report = await self.get_llm("researcher", "llm").invoke(
            prompts.research_prompt(question)
        )
        return f"## Web research\n{report}"

    async def _market_research(self, question: MetaculusQuestion) -> str:
        cheap = self.get_llm("parser", "llm")
        queries = parse_queries(await cheap.invoke(prompts.market_query_prompt(question)))
        if not queries:
            return ""
        quotes = (await asyncio.to_thread(markets.search_all, queries))[:MAX_MARKET_CANDIDATES]
        if not quotes:
            return ""
        verdict = await cheap.invoke(
            prompts.market_judge_prompt(question, markets.format_candidates(quotes))
        )
        if "no matches" in verdict.lower() and len(verdict) < 300:
            return ""
        found: list[MarketMatch] = await structure_output(
            verdict,
            list[MarketMatch],
            model=cheap,
            additional_instructions=(
                "Only include markets the text says are about the same event, using the index numbers "
                "from the text. Return an empty list if the text says nothing matches."
            ),
        )
        matches = [
            (quotes[item.index], item.match, item.note)
            for item in found
            if 0 <= item.index < len(quotes) and item.match in ("exact", "close")
        ]
        return markets.format_market_section(matches)

    ##################################### FORECASTING #####################################

    async def _make_prediction(
        self, question: MetaculusQuestion, research: str
    ) -> ReasonedPrediction:
        if isinstance(question, ConditionalQuestion):
            # The template forecasts each part of a conditional by calling back into
            # _make_prediction, so every part still gets the full ensemble.
            return await self._run_forecast_on_conditional(question, research)

        jobs = [spec for spec in self.forecasters for _ in range(spec.samples)]
        results = await asyncio.gather(
            *(self._forecast_once(question, research, spec) for spec in jobs),
            return_exceptions=True,
        )
        successes = [
            (spec, result)
            for spec, result in zip(jobs, results)
            if not isinstance(result, BaseException)
        ]
        failures = [
            f"{spec.short_name}: {type(result).__name__}: {str(result)[:300]}"
            for spec, result in zip(jobs, results)
            if isinstance(result, BaseException)
        ]
        labs_succeeded = {spec.lab for spec, _ in successes}
        labs_needed = min(self.min_labs, len({spec.lab for spec in self.forecasters}))
        if len(successes) < math.ceil(len(jobs) / 2) or len(labs_succeeded) < labs_needed:
            raise RuntimeError(
                f"Only {len(successes)} of {len(jobs)} forecasts succeeded across "
                f"{len(labs_succeeded)} labs for {question.page_url}. Failures: {failures}"
            )
        if failures:
            logger.warning(f"Some forecasts failed for {question.page_url}: {failures}")

        combined = self._combine(question, successes)
        return ReasonedPrediction(
            prediction_value=combined,
            reasoning=self._ensemble_reasoning(successes, combined, failures),
        )

    async def _forecast_once(
        self, question: MetaculusQuestion, research: str, spec: ForecasterSpec
    ) -> ReasonedPrediction:
        llm = llm_for(spec)
        note = self._get_conditional_disclaimer_if_necessary(question)
        if isinstance(question, BinaryQuestion):
            reasoning = await llm.invoke(prompts.binary_prompt(question, research, note))
            value = await self._parse_binary(reasoning)
        elif isinstance(question, MultipleChoiceQuestion):
            reasoning = await llm.invoke(
                prompts.multiple_choice_prompt(question, research, note)
            )
            value = await self._parse_multiple_choice(reasoning, question)
        elif isinstance(question, NumericQuestion):
            upper, lower = self._create_upper_and_lower_bound_messages(question)
            reasoning = await llm.invoke(
                prompts.numeric_prompt(question, research, lower, upper, note)
            )
            value = await self._parse_numeric(reasoning, question)
        elif isinstance(question, DateQuestion):
            upper, lower = self._create_upper_and_lower_bound_messages(question)
            reasoning = await llm.invoke(
                prompts.date_prompt(question, research, lower, upper, note)
            )
            value = await self._parse_date(reasoning, question)
        else:
            raise ValueError(f"Unsupported question type: {type(question).__name__}")
        logger.info(
            f"{spec.short_name} forecast {question.page_url}: {compact_prediction(value)}"
        )
        return ReasonedPrediction(prediction_value=value, reasoning=reasoning)

    ##################################### PARSING #####################################

    async def _parse_binary(self, reasoning: str) -> float:
        parsed: BinaryPrediction = await structure_output(
            reasoning,
            BinaryPrediction,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=clean_indents(
                f"""
                The text given to you is trying to give a probability forecast for a binary question.
                {self._create_resolved_question_parsing_message()}
                """
            ),
        )
        return max(0.01, min(0.99, parsed.prediction_in_decimal))

    async def _parse_multiple_choice(
        self, reasoning: str, question: MultipleChoiceQuestion
    ) -> PredictedOptionList:
        parsed: PredictedOptionList = await structure_output(
            text_to_structure=reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=clean_indents(
                f"""
                Make sure that all option names are one of the following:
                {question.options}

                The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
                Additionally, you may sometimes need to parse a 0% probability. Please do not skip options with 0% but rather make it an entry in your final list with 0% probability.
                {self._create_resolved_question_parsing_message()}
                """
            ),
        )
        return self._align_options(parsed, question)

    @staticmethod
    def _align_options(
        parsed: PredictedOptionList, question: MultipleChoiceQuestion
    ) -> PredictedOptionList:
        """Map parsed names onto the question's exact option names; fail the sample if any is missing."""
        by_name = {
            option.option_name.strip().casefold(): option.probability
            for option in parsed.predicted_options
        }
        missing = [name for name in question.options if name.strip().casefold() not in by_name]
        if missing:
            raise ValueError(f"Parsed forecast is missing options: {missing}")
        return PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name=name, probability=by_name[name.strip().casefold()])
                for name in question.options
            ]
        )

    async def _parse_numeric(
        self, reasoning: str, question: NumericQuestion
    ) -> NumericDistribution:
        percentiles: list[Percentile] = await structure_output(
            reasoning,
            list[Percentile],
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=clean_indents(
                f"""
                The text given to you is trying to give a forecast distribution for a numeric question.
                - This text is trying to answer the numeric question: "{question.question_text}".
                {self._create_single_distribution_parsing_message(question)}
                {self._create_resolved_question_parsing_message()}
                - When parsing the text, please make sure to give the values (the ones assigned to percentiles) in terms of the correct units.
                - The units for the forecast are: {question.unit_of_measure}
                - As an example, someone else guessed that the answer will be between {question.lower_bound} {question.unit_of_measure} and {question.upper_bound} {question.unit_of_measure}, so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
                - If the answer doesn't give the answer in the correct units, you should parse it in the right units. For instance if the answer gives numbers as $500,000,000 and units are "B $" then you should parse the answer as 0.5 (since $500,000,000 is $0.5 billion).
                - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
                - Turn any values that are in scientific notation into regular numbers.
                """
            ),
        )
        return NumericDistribution.from_question(percentiles, question)

    async def _parse_date(self, reasoning: str, question: DateQuestion) -> NumericDistribution:
        date_percentiles: list[DatePercentile] = await structure_output(
            reasoning,
            list[DatePercentile],
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=clean_indents(
                f"""
                The text given to you is trying to give a forecast distribution for a date question.
                - This text is trying to answer the question: "{question.question_text}".
                {self._create_single_distribution_parsing_message(question)}
                {self._create_resolved_question_parsing_message()}
                - The output is given as dates/times please format it into a valid datetime parsable string. Assume midnight UTC if no hour is given.
                - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
                """
            ),
        )
        percentiles = [
            Percentile(percentile=item.percentile, value=item.value.timestamp())
            for item in date_percentiles
        ]
        return NumericDistribution.from_question(percentiles, question)

    ##################################### COMBINING #####################################

    def _combine(self, question: MetaculusQuestion, successes: list[tuple[ForecasterSpec, ReasonedPrediction]]):
        labs = [spec.lab for spec, _ in successes]
        values = [result.prediction_value for _, result in successes]
        if isinstance(question, BinaryQuestion):
            return combine_binary(values, labs)
        if isinstance(question, MultipleChoiceQuestion):
            pooled = combine_multiple_choice(
                [
                    {option.option_name: option.probability for option in value.predicted_options}
                    for value in values
                ],
                labs,
            )
            return PredictedOptionList(
                predicted_options=[
                    PredictedOption(option_name=name, probability=probability)
                    for name, probability in pooled.items()
                ]
            )
        if isinstance(question, (NumericQuestion, DateQuestion)):
            cdfs = [value.get_cdf() for value in values]
            x_axis = [point.value for point in cdfs[0]]
            heights = combine_cdfs([[point.percentile for point in cdf] for cdf in cdfs], labs)
            return NumericDistribution.from_question(
                [Percentile(value=x, percentile=y) for x, y in zip(x_axis, heights)],
                question,
            )
        raise ValueError(f"Unsupported question type: {type(question).__name__}")

    @staticmethod
    def _ensemble_reasoning(
        successes: list[tuple[ForecasterSpec, ReasonedPrediction]],
        combined,
        failures: list[str],
    ) -> str:
        readable = compact_prediction
        lines = [
            "## How the final forecast was made",
            f"Final forecast: {readable(combined)}",
            "Each model forecast independently from the same research. Each AI lab gets equal weight: "
            "binary forecasts are averaged in log-odds, multiple choice and numeric forecasts as weighted averages.",
        ]
        for spec, result in successes:
            lines.append(f"- {spec.short_name}: {readable(result.prediction_value)}")
        if failures:
            lines.append(f"{len(failures)} forecast(s) failed and were left out.")
        for number, (spec, result) in enumerate(successes, start=1):
            lines.append(
                f"\n## Forecaster {number}: {spec.short_name} [{readable(result.prediction_value)}]\n{result.reasoning}"
            )
        return "\n".join(lines)
