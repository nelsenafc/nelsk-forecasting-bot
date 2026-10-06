import run


def test_minibench_waits_when_credit_runs_low():
    assert run.minibench_allowed(None)  # balance unreadable: carry on
    assert run.minibench_allowed(run.MINIBENCH_MIN_CREDIT_USD)
    assert not run.minibench_allowed(run.MINIBENCH_MIN_CREDIT_USD - 0.01)
