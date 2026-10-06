"""Prompt text for research, market matching and forecasting.

The forecasting prompts follow what has worked for strong bots in past seasons:
read the resolution criteria closely first, anchor on a base rate before the
news, then adjust, with prediction-market prices as a cross-check.
"""

from __future__ import annotations

from datetime import datetime, timezone

from forecasting_tools import MetaculusQuestion, MultipleChoiceQuestion, clean_indents


def _date(value: datetime | None) -> str:
    return value.strftime("%Y-%m-%d") if value else "not stated"


def today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def question_block(question: MetaculusQuestion) -> str:
    return clean_indents(
        f"""
        QUESTION
        {question.question_text}

        BACKGROUND
        {question.background_info or "None given."}

        RESOLUTION CRITERIA
        {question.resolution_criteria or "None given."}

        FINE PRINT
        {question.fine_print or "None given."}

        KEY DATES
        Today: {today()}
        Question closes for forecasting: {_date(question.close_time)}
        Scheduled resolution: {_date(question.scheduled_resolution_time)}
        """
    )


def research_prompt(question: MetaculusQuestion) -> str:
    return clean_indents(
        f"""
        You are the research assistant for a superforecaster working on the question below.
        Search the web and report the facts that matter for forecasting it. Do not give a forecast or a probability.

        {question_block(question)}

        Report the following, with a date and source name for every fact:
        1. Current status: the latest developments directly relevant to the resolution criteria.
        2. Already decided?: anything showing the outcome is already determined or impossible, judged strictly against the resolution criteria.
        3. Scheduled events before the resolution date that could decide the outcome.
        4. Base-rate material: historical frequencies, past values or comparable cases.
        5. Expectations: expert forecasts, polls, consensus estimates, prediction-market or betting prices.
        6. For quantities: the latest value, the recent trend and typical volatility over a comparable period.

        Be concise. Say clearly when something could not be found rather than guessing.
        """
    )


def market_query_prompt(question: MetaculusQuestion) -> str:
    return clean_indents(
        f"""
        Write 2 short search queries [3 to 6 words each] to find prediction markets about the same event as this forecasting question.
        Use the names, places and terms a market title would use. Output only the queries, one per line, nothing else.

        Question: {question.question_text}
        """
    )


def market_judge_prompt(question: MetaculusQuestion, candidates: str) -> str:
    return clean_indents(
        f"""
        You are checking which prediction markets are about the same thing as a forecasting question.

        {question_block(question)}

        CANDIDATE MARKETS
        {candidates}

        For every candidate that concerns the same event, give:
        - its index number
        - how closely it matches: "exact" [same event, deadline and criteria], "close" [same event with small differences in deadline, threshold or source] or "related" [informative, but a different event or question]
        - one sentence on how it differs from the question, including deadline differences

        Leave out candidates about different events. If none match, say "no matches".
        """
    )


def price_spec_prompt(question: MetaculusQuestion) -> str:
    return clean_indents(
        f"""
        Decide whether this forecasting question resolves on a single observation of a market price or published data series that a free data feed carries, and if so, which one.

        {question_block(question)}

        UNITS: {question.unit_of_measure or "not stated"}

        Supported feeds:
        - yahoo: Yahoo Finance daily closes. Stocks and ETFs by ticker [FXI, SPY, AAPL], indices [^GSPC S&P 500, ^IXIC Nasdaq Composite, ^DJI Dow, ^VIX, ^N225 Nikkei, ^HSI Hang Seng, ^STOXX50E], front-month futures [BZ=F Brent, CL=F WTI, NG=F natural gas, GC=F gold, SI=F silver, HG=F copper, RB=F RBOB gasoline, ZC=F corn, ZW=F wheat], FX [EURUSD=X, JPY=X for USD/JPY, CNY=X for USD/CNY, GBPUSD=X], crypto [BTC-USD, ETH-USD, SOL-USD].
        - fred: FRED series by ID [GASREGW US regular gasoline weekly from EIA, DGS10 10-year Treasury yield, DGS2, DFF fed funds effective rate, DCOILBRENTEU Brent spot, DCOILWTICO WTI spot, DEXUSEU USD per EUR, T10Y2Y].
        - fear_greed: the alternative.me Crypto Fear & Greed Index, 0 to 100.

        Answer is_price_question = true only if ALL of these hold:
        - the answer is one value of the series for one date, such as a close, a settlement or a weekly reading;
        - the feed tracks the same instrument or statistic the question resolves on. A different publisher's figure for a similar quantity does not count [the AAA daily gas price is not the EIA weekly series]. Small definitional differences on the same instrument are fine: for futures, Yahoo's daily close is the exchange settlement price, and a close versus an official close or last trade does not matter;
        - it is not a maximum, minimum, average or count over a period.

        If true, also give:
        - source and symbol from the lists above [another ticker or FRED ID is fine if it is the exact series];
        - target_date: YYYY-MM-DD of the observation that resolves the question;
        - kind: "log" for prices and anything that moves in percentage terms; "level" for bounded indices like Fear & Greed and for interest rates in percent;
        - multiplier: the number that converts the feed's units into the question's units, usually 1;
        - note: one sentence on any mismatch to watch.
        If false, set source to "none" and explain in the note.
        """
    )


_STEPS_INTRO = "Work through these steps in order and write each one out:"

_RESOLUTION_STEP = (
    "1. Resolution check. Restate precisely what must happen, by what deadline, and according to which source. "
    "Flag edge cases in the fine print. Check whether the research shows the outcome is already determined. "
    "Do not assume an event has happened unless the research clearly shows it, and do not treat the question as already resolved."
)

_EVIDENCE_STEP = (
    "4. Evidence. List the evidence that should move you away from the base rate, with direction and rough strength. "
    "Give observed actions more weight than statements or promises, and recent reliable sources more weight than old or partisan ones."
)

_MARKET_STEP = (
    "5. Market and expert check. If the research shows a prediction-market price or expert forecast for a closely matching question, "
    "state it and check that its resolution criteria really match. A liquid real-money market is hard to beat without a specific reason, "
    "so explain any disagreement. Play-money markets count for much less."
)


def binary_prompt(question: MetaculusQuestion, research: str, conditional_note: str) -> str:
    return clean_indents(
        f"""
        You are a superforecaster with a strong Metaculus track record. Forecast the probability that this question resolves YES.

        {question_block(question)}

        RESEARCH [gathered just now; may contain errors or irrelevant items]
        {research}

        {_STEPS_INTRO}
        {_RESOLUTION_STEP}
        2. Time left. How long until resolution, and which scheduled events fall inside that window.
        3. Base rate. Name the most relevant reference class and estimate how often outcomes like this happen, before using the news. Show the numbers.
        {_EVIDENCE_STEP}
        {_MARKET_STEP}
        6. Final answer. Start from the base rate and adjust for the evidence. The world changes slowly most of the time, so give the status quo extra weight. Be decisive when the evidence is strong, and avoid both overconfidence and drifting toward 50% out of caution.

        {conditional_note}

        The last line of your answer must be: "Probability: ZZ%" with ZZ between 1 and 99.
        """
    )


def multiple_choice_prompt(
    question: MultipleChoiceQuestion, research: str, conditional_note: str
) -> str:
    return clean_indents(
        f"""
        You are a superforecaster with a strong Metaculus track record. Forecast the probability of each option.

        {question_block(question)}

        OPTIONS
        {question.options}

        RESEARCH [gathered just now; may contain errors or irrelevant items]
        {research}

        {_STEPS_INTRO}
        {_RESOLUTION_STEP}
        2. Time left. How long until resolution, and which scheduled events fall inside that window.
        3. Base rates. For the leading options, estimate how often outcomes like each happen in comparable cases, before using the news.
        {_EVIDENCE_STEP}
        {_MARKET_STEP}
        6. Final answer. Give the status quo extra weight. Leave some probability on every option for surprises, but do not spread probability evenly out of caution.

        {conditional_note}

        End with your final probabilities for the options in this exact order {question.options}, one per line:
        Option_A: Probability_A
        Option_B: Probability_B
        ...
        Option_N: Probability_N
        """
    )


_PERCENTILE_LINES = "\n".join(
    f"Percentile {p}: XX" for p in (5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95)
)

_DATE_PERCENTILE_LINES = "\n".join(
    f"Percentile {p}: YYYY-MM-DD" for p in (5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95)
)


def numeric_prompt(
    question: MetaculusQuestion,
    research: str,
    lower_bound_message: str,
    upper_bound_message: str,
    conditional_note: str,
) -> str:
    units = question.unit_of_measure or "Not stated [infer it from the question]"
    return clean_indents(
        f"""
        You are a superforecaster with a strong Metaculus track record. Forecast a probability distribution for this quantity.

        {question_block(question)}

        UNITS FOR THE ANSWER: {units}
        {lower_bound_message}
        {upper_bound_message}

        RESEARCH [gathered just now; may contain errors or irrelevant items]
        {research}

        {_STEPS_INTRO}
        1. Resolution check. Restate exactly which quantity is measured, by which source, on what date, in which units.
        2. Latest value and trend. The most recent value in the research and how it has been moving.
        3. Base rate. How much has this quantity typically moved over a period as long as the time left? Use history to size your spread.
        4. Scheduled events and evidence that could push it up or down.
        5. Expert and market expectations, if any, and whether they match this exact quantity.
        6. Distribution. Set your median first, then the spread. Keep the center sharp when the data is strong, and the tails wide enough for surprises.

        {conditional_note}

        Formatting: give values in the requested units, never use scientific notation, and make values increase with the percentile.
        End with your final answer exactly as:
        {_PERCENTILE_LINES}
        """
    )


def date_prompt(
    question: MetaculusQuestion,
    research: str,
    lower_bound_message: str,
    upper_bound_message: str,
    conditional_note: str,
) -> str:
    return clean_indents(
        f"""
        You are a superforecaster with a strong Metaculus track record. Forecast a probability distribution for when this will happen.

        {question_block(question)}

        {lower_bound_message}
        {upper_bound_message}

        RESEARCH [gathered just now; may contain errors or irrelevant items]
        {research}

        {_STEPS_INTRO}
        1. Resolution check. Restate exactly which event counts, by which source.
        2. Current status and the remaining steps before the event can happen.
        3. Base rate. How long comparable processes have taken historically.
        4. Scheduled events and evidence that could speed it up or delay it.
        5. Expert and market expectations, if any.
        6. Distribution. Set your median date first, then the spread, with tails wide enough for surprises.

        {conditional_note}

        Formatting: dates as YYYY-MM-DD in chronological order, earliest at percentile 5.
        End with your final answer exactly as:
        {_DATE_PERCENTILE_LINES}
        """
    )
