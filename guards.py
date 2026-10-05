"""Rules the bot keeps about which questions it may be tried on.

Kept free of side effects [no .env loading] so tests can import it safely.
"""

from __future__ import annotations

from forecasting_tools import MetaculusClient, MetaculusQuestion
from forecasting_tools.data_models.questions import QuestionState

LIVE_TOURNAMENT_MARKERS = ("futureeval", "minibench", "aib")


def is_live_bot_tournament(question: MetaculusQuestion) -> bool:
    if question.default_project_id == MetaculusClient.FE_FALL_2026_ID:
        return True
    slugs = [slug.lower() for slug in question.tournament_slugs or []]
    return any(marker in slug for slug in slugs for marker in LIVE_TOURNAMENT_MARKERS)


def may_preview(question: MetaculusQuestion) -> bool:
    """Tournament rules allow testing on bot-tournament questions only once they have closed."""
    if not is_live_bot_tournament(question):
        return True
    return question.state in (QuestionState.CLOSED, QuestionState.RESOLVED)
