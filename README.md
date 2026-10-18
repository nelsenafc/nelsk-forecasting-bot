# Metaculus forecasting bot

A bot for the [Metaculus FutureEval](https://www.metaculus.com/futureeval/) bot tournaments, built on the Metaculus [template bot](https://github.com/Metaculus/metac-bot-template) and its `forecasting-tools` package. The upstream README is kept as `TEMPLATE_README.md`.

## How it forecasts

1. **Research, once per question.** Three sources run in parallel:
   - AskNews articles [free for tournament bots]
   - a web-search model [GPT-6.1 Sol with the provider's native search] asked for current status, scheduled events, base-rate material and expert or market expectations
   - prediction markets on Polymarket [real money] and Manifold [mostly play money], where a cheap model first judges whether each market is about the same event and how its resolution differs
   - the pages the question says it resolves on, read through a rendering reader [r.jina.ai, free] with a plain fetch as backup, then a cheap model notes the current value and its date. The web-search model often cannot open these pages: on 37 closed questions its research said 19 times that it could not open or verify something
   - how similar questions the bot forecast before resolved, with its forecast and score on each. 34% of Fall 2025 winners looked up similar resolved questions, against 0% of non-winners. Metaculus hides outcomes from a bot account unless it forecast the question, so this draws on the bot's own history, kept on the `bot-data` branch
2. **Forecast with three labs.** The same research goes to GPT-6.1 Sol [2 samples], Claude Sonnet 5.5 [1] and Gemini 3.8 Flash [2]. Fall 2026 questions carry the prize money, about 20x a MiniBench question each, so they get one more sample per lab and Claude at high effort. Each one checks the resolution criteria, anchors on a base rate before reading the news, weighs the evidence, cross-checks any matching market, then answers.
3. **Combine.** Each lab gets equal weight however many samples it ran. Binary forecasts are averaged in log-odds, multiple choice as a weighted average with a small floor on every option, numeric and date questions as a mixture of the CDFs.
4. **Price questions get a statistical model too.** When a numeric question resolves on one value of a market series on one date [a stock or ETF close, a futures settlement, an FX rate, crypto, a FRED series, the crypto Fear & Greed index], a cheap model maps it to a free data feed and `price_model.py` simulates where the series lands: every past window as long as the horizon, standardised by the volatility of its day and rescaled to today's. The forecasters see that distribution in their research, and the final forecast blends it 50/50 with the AI ensemble [`PRICE_MODEL_WEIGHT`]. It is skipped when the feed is stale, when the series' latest value sits far outside the question's range, or when the question's figure comes from a different publisher.
5. **Publish** the forecast with a private comment holding every model's reasoning, as the tournament rules require.

A question is skipped, and retried on the next run, if both news and web research fail or if fewer than half the forecasts succeed across at least two labs.

## Files

| File | What it does |
|---|---|
| `forecaster.py` | The bot: research, the ensemble, parsing |
| `ensemble.py` | The forecaster line-up and the combining math |
| `markets.py` | Polymarket and Manifold lookups |
| `resolution_sources.py` | Reads the pages each question says it resolves on |
| `history.py` | The bot's own resolved questions, for the similar-questions lookup |
| `agent.py` | The agent candidate: researches with tools in a loop, then forecasts |
| `shadow.py` | Runs the agent on questions right after they close, never publishing |
| `bot_data.py` | The hourly data job: history and the shadow test |
| `calibration.py` | Platt scaling and caps, judged on held-out questions before going live |
| `price_model.py` | Statistical forecasts for questions about a market series on a date |
| `backtest_price_model.py` | Scores the price model on five years of history it never saw |
| `forecast_log.py` | Exports the bot's tournament forecasts, values only after each question closes |
| `notion_sync.py` | Keeps a Notion copy of that log current, hourly on GitHub |
| `scorecard.py` | Scores resolved forecasts: peer score against the field, calibration, which labs earn their cost, and the price model against the AI ensemble |
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

GitHub Actions runs `--mode tournament`. Repository secrets needed: `METACULUS_TOKEN`, `OPENROUTER_API_KEY`, `ASKNEWS_API_KEY`.

Tournament questions stay open for 3 hours, and GitHub's scheduler drops most runs on a small repo [2 of 33 fired on 5-6 Oct 2026]. So an outside scheduler starts the workflow every 20 minutes through the GitHub API: a launchd job running `gh workflow run` on a Mac covers the hours it is awake, and cron-job.org [free] covers the rest. The workflow's own 5-minute schedule is the fallback. The cron-job.org call:

- `POST https://api.github.com/repos/nelsenafc/nelsk-forecasting-bot/actions/workflows/run_bot_on_tournament.yaml/dispatches` with body `{"ref":"main"}`
- headers `Authorization: Bearer <token>` and `Accept: application/vnd.github+json`, where the token is fine-grained, limited to this repo, with Actions read and write and nothing else

A second workflow runs `notion_sync.py` hourly once a `NOTION_TOKEN` secret exists [a Notion internal integration connected to the log page]. It adds each new forecast as a row, fills in the numbers after the question closes and the outcome and peer score after it resolves, and prints counts only.

## The agent candidate

In MiniBench round 1 [Oct 2026] Metaculus ran the same models one-shot and as agents that search, read pages and search again before forecasting. The agent won on every model, by about 7 peer points a question at the median. Gemini 3.8 Flash, a cheap model, scored +12.6 as an agent and +5.3 one-shot. This bot is one-shot, so `agent.py` builds the candidate: Gemini 3.8 Flash with four tools [web search, page reader, prediction markets, the bot's resolved history], then a forecast from its own brief. A research run costs about $0.16.

It is tested before it goes live. Each hour the data job runs it on tournament questions the live bot forecast, within hours of their closing, and publishes nothing, so no one sees a forecast on an open question and the live forecast stays the only one submitted. The scorecard pairs the two on every resolved question. The live bot switches when the agent leads by 3 points a question or more over 40 or more questions.

## Calibration

`calibration.py` fits Platt scaling to the bot's resolved yes/no forecasts and tests floors and ceilings, each scored only on questions it was not fitted on. Metaculus found Platt scaling the best post-hoc correction for bots, by a small margin, and capping the strongest difference between winning bots. A correction goes live through `calibration.json` once it wins on 50 or more held-out questions. Until then the bot's forecasts pass through unchanged. A guessed correction is worse than none: pushing forecasts away from 50% by a factor of 1.3 would have cost 34.5 points on the first 7 resolved yes/no questions.

## Price model backtest

`python backtest_price_model.py` replays the model on ten series [FXI, Brent, WTI, S&P 500, gold, EUR/USD, Bitcoin, VIX, US weekly gas, crypto Fear & Greed] at horizons of one to four weeks. Across all of them the 80% range contained the outcome 80% of the time and the 50% range 52%, and its pinball loss beat a constant-volatility normal by 1.1% on average [5-7% on the S&P 500, VIX and Fear & Greed].

## Rules it keeps

- No human in the loop. Dry mode refuses tournament questions that are still open or upcoming, so nobody previews a forecast before it counts; closed ones are fair game for testing.
- One forecast per question: previously forecast questions are skipped.
- Every forecast carries a comment with the reasoning behind it.
- The run stops when the OpenRouter key drops below `MIN_CREDIT_USD` [default $2], instead of failing halfway through a question.
- Below `MINIBENCH_MIN_CREDIT_USD` [default $6], MiniBench waits so the remaining credit goes to Fall 2026.
