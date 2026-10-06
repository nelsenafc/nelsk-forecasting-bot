"""Run the forecasting bot.

Modes
  tournament  New questions in the Fall 2026 FutureEval tournament and the current
              MiniBench. Publishes. This is what the scheduled GitHub workflow runs.
  test        A few questions from the unscored bot-testing-area. Publishes there
              only, so it is a safe end-to-end check of keys, research and submission.
  cup         The Metaculus Cup, for practice against the human crowd. Publishes.
  dry         One question by URL, nothing published. Refuses bot-tournament questions
              that are still open or upcoming: previewing those breaks the
              no-human-in-the-loop rule. Closed ones are fair game for testing.

Examples
  python run.py --mode test --limit 3
  python run.py --mode dry --question-url https://www.metaculus.com/questions/12345/
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from bot_helpers import (
    check_environment,
    print_run_summary_banner,
    print_startup_banner,
    silence_noisy_dependencies,
)

silence_noisy_dependencies()

import dotenv
import requests
from forecasting_tools import MetaculusClient, MetaculusQuestion

from forecaster import EnsembleBot
from guards import may_preview

dotenv.load_dotenv()
logger = logging.getLogger(__name__)

TEST_TOURNAMENT = "bot-testing-area"
MIN_CREDIT_USD = float(os.getenv("MIN_CREDIT_USD", "2"))
TOURNAMENT_URLS = {
    "tournament": "https://www.metaculus.com/tournament/fall-futureeval-2026/",
    "cup": "https://www.metaculus.com/tournament/metaculus-cup-fall-2026/",
    "test": "https://www.metaculus.com/tournament/bot-testing-area/",
}


def openrouter_credit_left() -> float | None:
    """Remaining credit on the OpenRouter key, or None if it can't be read."""
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        return None
    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/key",
            headers={"Authorization": f"Bearer {key}"},
            timeout=15,
        )
        response.raise_for_status()
        remaining = response.json().get("data", {}).get("limit_remaining")
        return float(remaining) if remaining is not None else None
    except (requests.RequestException, ValueError) as error:
        logger.warning(f"Could not read OpenRouter credit balance: {error}")
        return None


def pick_varied(questions: list[MetaculusQuestion], limit: int) -> list[MetaculusQuestion]:
    """Take up to `limit` questions, cycling through question types so a test covers each."""
    by_type: dict[str, list[MetaculusQuestion]] = {}
    for question in questions:
        by_type.setdefault(type(question).__name__, []).append(question)
    picked: list[MetaculusQuestion] = []
    while len(picked) < limit and any(by_type.values()):
        for bucket in by_type.values():
            if bucket and len(picked) < limit:
                picked.append(bucket.pop(0))
    return picked


def build_bot(publish: bool, skip_previous: bool, use_markets: bool) -> EnsembleBot:
    return EnsembleBot(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=publish,
        folder_to_save_reports_to="reports",
        skip_previously_forecasted_questions=skip_previous,
        extra_metadata_in_explanation=True,
        use_market_signals=use_markets,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ensemble forecasting bot")
    parser.add_argument("--mode", choices=["tournament", "test", "cup", "dry"], default="tournament")
    parser.add_argument(
        "--question-url", action="append", default=[], help="Question to forecast in dry mode [repeatable]"
    )
    parser.add_argument("--limit", type=int, default=3, help="Questions to forecast in test mode")
    parser.add_argument("--no-markets", action="store_true", help="Skip the prediction-market lookup")
    args = parser.parse_args()

    # The repo is public, so its GitHub Actions logs are too. In tournament mode the logs
    # must not show forecasts or reasoning while questions are still open to other bots.
    quiet = args.mode == "tournament"
    logging.basicConfig(
        level=logging.WARNING if quiet else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    check_environment(strict=True)
    credit = openrouter_credit_left()
    if credit is not None:
        print(f"OpenRouter credit left: ${credit:.2f}")
        if credit < MIN_CREDIT_USD:
            print(f"⛔  Only ${credit:.2f} of OpenRouter credit left [minimum ${MIN_CREDIT_USD:.2f}]. Not forecasting.")
            return

    client = MetaculusClient()
    use_markets = not args.no_markets
    publish = args.mode != "dry"
    print_startup_banner(args.mode, will_publish=publish)

    if args.mode == "tournament":
        bot = build_bot(publish=True, skip_previous=True, use_markets=use_markets)
        reports = asyncio.run(
            bot.forecast_on_tournament(client.CURRENT_AI_COMPETITION_ID, return_exceptions=True)
        )
        reports += asyncio.run(
            bot.forecast_on_tournament(client.CURRENT_MINIBENCH_ID, return_exceptions=True)
        )
    elif args.mode == "test":
        bot = build_bot(publish=True, skip_previous=False, use_markets=use_markets)
        questions = pick_varied(
            client.get_all_open_questions_from_tournament(TEST_TOURNAMENT), args.limit
        )
        reports = asyncio.run(bot.forecast_questions(questions, return_exceptions=True))
    elif args.mode == "cup":
        bot = build_bot(publish=True, skip_previous=True, use_markets=use_markets)
        reports = asyncio.run(
            bot.forecast_on_tournament(client.CURRENT_METACULUS_CUP_ID, return_exceptions=True)
        )
    else:
        if not args.question_url:
            sys.exit("Dry mode needs at least one --question-url")
        questions = [client.get_question_by_url(url) for url in args.question_url]
        for question in questions:
            if not may_preview(question):
                sys.exit(
                    f"Refusing: {question.page_url} is a bot-tournament question that is still open or upcoming. "
                    "Previewing the bot's forecast on it would break the no-human-in-the-loop rule. Use a closed "
                    "tournament question, a bot-testing-area question or a main-site question."
                )
        bot = build_bot(publish=False, skip_previous=False, use_markets=use_markets)
        reports = asyncio.run(bot.forecast_questions(questions, return_exceptions=True))

    if not quiet:
        bot.log_report_summary(reports)
    costs = [report.price_estimate or 0 for report in reports if not isinstance(report, BaseException)]
    if costs:
        print(f"Model cost this run: ${sum(costs):.2f} over {len(costs)} question(s)")
    if publish and os.getenv("GITHUB_OUTPUT"):
        # lets the workflow start the Notion log sync only after a run that posted forecasts
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            output.write(f"forecasts={len(costs)}\n")
    print_run_summary_banner(reports, will_publish=publish, tournament_url=TOURNAMENT_URLS.get(args.mode))


if __name__ == "__main__":
    main()
