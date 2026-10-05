# Metaculus forecasting bot

A bot for the [Metaculus FutureEval](https://www.metaculus.com/futureeval/) bot tournaments, built on the Metaculus [template bot](https://github.com/Metaculus/metac-bot-template) and its `forecasting-tools` package. The upstream README is kept as `TEMPLATE_README.md`.

## How it forecasts

1. **Research, once per question.** Three sources run in parallel:
   - AskNews articles [free for tournament bots]
   - a web-search model [GPT-6.1 Sol with the provider's native search] asked for current status, scheduled events, base-rate material and expert or market expectations
   - prediction markets on Polymarket [real money] and Manifold [mostly play money], where a cheap model first judges whether each market is about the same event and how its resolution differs
2. **Forecast with three labs.** The same research goes to GPT-6.1 Sol [2 samples], Claude Sonnet 5.5 [1] and Gemini 3.8 Flash [2]. Each one checks the resolution criteria, anchors on a base rate before reading the news, weighs the evidence, cross-checks any matching market, then answers.
3. **Combine.** Each lab gets equal weight however many samples it ran. Binary forecasts are averaged in log-odds, multiple choice as a weighted average with a small floor on every option, numeric and date questions as a mixture of the CDFs.
4. **Publish** the forecast with a private comment holding every model's reasoning, as the tournament rules require.

A question is skipped, and retried on the next run, if both news and web research fail or if fewer than half the forecasts succeed across at least two labs.

## Files

| File | What it does |
|---|---|
| `forecaster.py` | The bot: research, the ensemble, parsing |
| `ensemble.py` | The forecaster line-up and the combining math |
| `markets.py` | Polymarket and Manifold lookups |
| `prompts.py` | Research, market-matching and forecasting prompts |
| `run.py` | Entry point and run modes |
| `tests/` | Offline tests, including a full pipeline run with fake models |
| `main.py`, `main_with_no_framework.py` | The upstream template, kept for reference |

## Running it

```bash
cp .env.template .env   # then fill in the keys
pip install poetry && poetry install
poetry run python -m pytest                              # offline tests
poetry run python run.py --mode test --limit 3           # unscored bot-testing-area, publishes there
poetry run python run.py --mode dry --question-url URL   # one question, nothing published
poetry run python run.py --mode tournament               # live tournament + MiniBench
```

GitHub Actions runs `--mode tournament` every 20 minutes. Repository secrets needed: `METACULUS_TOKEN`, `OPENROUTER_API_KEY`, `ASKNEWS_API_KEY`.

## Rules it keeps

- No human in the loop. Dry mode refuses questions from live bot tournaments, so nobody previews a forecast before it counts.
- One forecast per question: previously forecast questions are skipped.
- Every forecast carries a comment with the reasoning behind it.
- The run stops when the OpenRouter key drops below `MIN_CREDIT_USD` [default $2], instead of failing halfway through a question.
