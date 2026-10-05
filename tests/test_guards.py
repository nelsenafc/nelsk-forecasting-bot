from forecasting_tools import BinaryQuestion, MetaculusClient
from forecasting_tools.data_models.questions import QuestionState

from guards import may_preview


def make(state, slugs=(), project=None):
    return BinaryQuestion(
        question_text="Will it happen?",
        state=state,
        tournament_slugs=list(slugs),
        default_project_id=project,
    )


def test_open_or_upcoming_tournament_questions_are_off_limits():
    assert not may_preview(make(QuestionState.OPEN, ["minibench-2026-10-05"]))
    assert not may_preview(make(QuestionState.UPCOMING, ["fall-futureeval-2026"]))
    assert not may_preview(make(QuestionState.OPEN, project=MetaculusClient.FE_FALL_2026_ID))


def test_closed_tournament_questions_can_be_tested():
    assert may_preview(make(QuestionState.CLOSED, ["minibench-2026-10-05"]))
    assert may_preview(make(QuestionState.RESOLVED, ["fall-futureeval-2026"]))


def test_practice_and_main_site_questions_are_fine():
    assert may_preview(make(QuestionState.OPEN, ["bot-testing-area"]))
    assert may_preview(make(QuestionState.OPEN, ["metaculus-cup-fall-2026"]))
