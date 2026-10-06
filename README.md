# Metaculus forecasting bot

A bot for the [Metaculus FutureEval](https://www.metaculus.com/futureeval/) bot tournaments, built on the Metaculus [template bot](https://github.com/Metaculus/metac-bot-template) and its `forecasting-tools` package. The upstream README is kept as `TEMPLATE_README.md`.

## How it forecasts

1. **Research, once per question.** Three sources run in parallel:
   - AskNews articles [free for tournament bots]
   - a web-search model [GPT-6.1 Sol with the provider's native search] asked for current status, scheduled events, base-rate material and expert or market expectations
   - prediction markets on Polymarket [real money] and Manifold [mostly play money], where a cheap model first judges whether each market is about the same event and how its resolution differs
2. **Forecast with three labs.** The same research goes to GPT-6.1 Sol [2 samples], Claude Sonnet 5.5 [1] and Gemini 3.8 Flash [2]. Each one checks the resolution criteria, anchors on a base rate before reading the news, weighs the evidence, cross-checks any matching market, then answers.
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
| `price_model.py` | Statistical forecasts for questions about a market series on a date |
| `backtest_price_model.py` | Scores the price model on five years of history it never saw |
| `forecast_log.py` | Exports the bot's tournament forecasts, values only after each question closes |
| `notion_sync.py` | Keeps a Notion copy of that log current, hourly on GitHub |
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

## Price model backtest

`python backtest_price_model.py` replays the model on ten series [FXI, Brent, WTI, S&P 500, gold, EUR/USD, Bitcoin, VIX, US weekly gas, crypto Fear & Greed] at horizons of one to four weeks. Across all of them the 80% range contained the outcome 80% of the time and the 50% range 52%, and its pinball loss beat a constant-volatility normal by 1.1% on average [5-7% on the S&P 500, VIX and Fear & Greed].

## Rules it keeps

- No human in the loop. Dry mode refuses tournament questions that are still open or upcoming, so nobody previews a forecast before it counts; closed ones are fair game for testing.
- One forecast per question: previously forecast questions are skipped.
- Every forecast carries a comment with the reasoning behind it.
- The run stops when the OpenRouter key drops below `MIN_CREDIT_USD` [default $2], instead of failing halfway through a question.
