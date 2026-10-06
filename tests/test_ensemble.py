import math

import pytest

from ensemble import (
    combine_binary,
    combine_cdfs,
    combine_multiple_choice,
    default_forecasters,
    fall_forecasters,
    lab_weights,
    logit,
)


def test_fall_line_up_adds_a_sample_per_lab_and_claude_thinks_harder():
    default = {spec.lab: spec for spec in default_forecasters()}
    fall = {spec.lab: spec for spec in fall_forecasters()}
    assert {lab: spec.samples for lab, spec in fall.items()} == {
        lab: spec.samples + 1 for lab, spec in default.items()
    }
    assert fall["anthropic"].reasoning_effort == "high"
    assert fall["openai"].reasoning_effort == default["openai"].reasoning_effort
    assert [spec.model for spec in fall_forecasters()] == [spec.model for spec in default_forecasters()]


def test_each_lab_gets_equal_total_weight():
    labs = ["openai", "openai", "anthropic", "google", "google"]
    weights = lab_weights(labs)
    totals = {}
    for lab, weight in zip(labs, weights):
        totals[lab] = totals.get(lab, 0) + weight
    assert all(math.isclose(total, 1 / 3) for total in totals.values())
    assert math.isclose(sum(weights), 1.0)


def test_binary_agreement_is_preserved():
    assert math.isclose(combine_binary([0.3, 0.3, 0.3], ["a", "b", "c"]), 0.3)


def test_binary_pools_in_log_odds():
    combined = combine_binary([0.9, 0.5], ["a", "b"])
    assert math.isclose(logit(combined), logit(0.9) / 2)


def test_binary_lab_with_more_samples_does_not_dominate():
    # Two OpenAI samples at 90% and one Anthropic sample at 10% balance out to 50%.
    combined = combine_binary([0.9, 0.9, 0.1], ["openai", "openai", "anthropic"])
    assert math.isclose(combined, 0.5)


def test_binary_is_clipped_and_survives_extremes():
    assert combine_binary([1.0, 1.0], ["a", "b"]) == 0.99
    assert combine_binary([0.0, 0.0], ["a", "b"]) == 0.01


def test_binary_needs_matching_lists():
    with pytest.raises(ValueError):
        combine_binary([0.5], ["a", "b"])


def test_multiple_choice_sums_to_one_with_floor():
    pooled = combine_multiple_choice(
        [{"A": 0.7, "B": 0.3, "C": 0.0}, {"A": 0.5, "B": 0.5, "C": 0.0}],
        ["a", "b"],
    )
    assert math.isclose(sum(pooled.values()), 1.0)
    assert pooled["C"] > 0.004
    assert pooled["A"] > pooled["B"]


def test_multiple_choice_normalises_each_forecast_first():
    pooled = combine_multiple_choice([{"A": 60, "B": 40}], ["a"], floor=0)
    assert math.isclose(pooled["A"], 0.6)


def test_multiple_choice_rejects_mismatched_options():
    with pytest.raises(ValueError):
        combine_multiple_choice([{"A": 0.5, "B": 0.5}, {"A": 0.5, "C": 0.5}], ["a", "b"])


def test_cdfs_average_pointwise_by_lab():
    combined = combine_cdfs([[0.0, 0.2, 1.0], [0.0, 0.6, 1.0], [0.0, 0.4, 1.0]], ["a", "a", "b"])
    # Lab a averages to 0.4, lab b is 0.4.
    assert combined == pytest.approx([0.0, 0.4, 1.0])


def test_cdfs_must_share_axis():
    with pytest.raises(ValueError):
        combine_cdfs([[0.0, 1.0], [0.0, 0.5, 1.0]], ["a", "b"])
