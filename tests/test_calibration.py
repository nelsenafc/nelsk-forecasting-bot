import math
import random

import pytest

import calibration


def synthetic(n, true_a, seed=7):
    """Forecasts that are too timid: what happens follows sigmoid(true_a * logit(p))."""
    rng = random.Random(seed)
    pairs = []
    for _ in range(n):
        p = rng.uniform(0.05, 0.95)
        truth = 1 / (1 + math.exp(-true_a * math.log(p / (1 - p))))
        pairs.append((p, rng.random() < truth))
    return pairs


def test_fit_recovers_timidity_from_plenty_of_data():
    a, b = calibration.fit(synthetic(4000, true_a=1.6))
    assert a == pytest.approx(1.6, abs=0.15)
    assert b == pytest.approx(0.0, abs=0.1)


def test_fit_stays_near_no_change_on_a_handful_of_questions():
    a, b = calibration.fit([(0.09, True), (0.10, False), (0.65, True)])
    assert abs(a - 1) < 0.5 and abs(b) < 0.5


def test_leave_one_out_rewards_a_real_effect_and_caps_are_scored():
    pairs = synthetic(300, true_a=1.8)
    assert sum(calibration.leave_one_out(pairs)) / len(pairs) > 0
    gains = calibration.cap_gains([(0.02, True), (0.5, False)], 0.05, 0.95)
    assert gains[0] > 0 and gains[1] == 0


def test_nothing_changes_without_a_config(monkeypatch, tmp_path):
    monkeypatch.setattr(calibration, "CONFIG", tmp_path / "missing.json")
    assert calibration.apply_binary(0.07) == 0.07
    (tmp_path / "on.json").write_text('{"binary": {"a": 1.0, "b": 0.0, "floor": 0.05, "ceiling": 0.95}}')
    monkeypatch.setattr(calibration, "CONFIG", tmp_path / "on.json")
    assert calibration.apply_binary(0.02) == pytest.approx(0.05)
