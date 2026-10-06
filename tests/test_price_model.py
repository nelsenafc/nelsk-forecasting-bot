from datetime import date, timedelta

import numpy as np
import pytest
from forecasting_tools import NumericQuestion

import price_model
from forecaster import price_model_problem


def random_walk(n=1500, daily_vol=0.02, start=100.0, seed=1):
    rng = np.random.default_rng(seed)
    return start * np.exp(np.cumsum(rng.normal(0, daily_vol, n)))


def business_days(n, end=date(2026, 10, 2)):
    days, current = [], end
    while len(days) < n:
        if current.weekday() < 5:
            days.append(current)
        current -= timedelta(days=1)
    return list(reversed(days))


def test_ewma_tracks_a_known_volatility():
    moves = np.random.default_rng(0).normal(0, 0.02, 2000)
    assert price_model.ewma_volatility(moves)[-1] == pytest.approx(0.02, rel=0.35)


def test_steps_count_trading_days_calendar_days_and_weeks():
    friday = date(2026, 10, 2)
    assert price_model.steps_until(friday, date(2026, 10, 9), "business") == 5
    assert price_model.steps_until(friday, date(2026, 10, 9), "daily") == 7
    assert price_model.steps_until(friday, date(2026, 10, 16), "weekly") == 2
    assert price_model.steps_until(friday, friday, "business") == 0


def test_frequency_is_read_from_the_dates():
    assert price_model.frequency(business_days(100)) == "business"
    daily = [date(2026, 6, 1) + timedelta(days=i) for i in range(100)]
    assert price_model.frequency(daily) == "daily"
    weekly = [date(2025, 1, 6) + timedelta(weeks=i) for i in range(100)]
    assert price_model.frequency(weekly) == "weekly"


def test_random_walk_spread_scales_with_the_square_root_of_time():
    values = random_walk()
    terminal = price_model.simulate_direct(values, 9, n_paths=20000)
    spread = np.std(np.log(terminal / values[-1]))
    assert spread == pytest.approx(0.02 * 3, rel=0.35)
    assert np.median(terminal) == pytest.approx(values[-1], rel=0.03)


def test_whole_windows_narrow_the_spread_for_a_series_that_pulls_back():
    rng = np.random.default_rng(2)
    level = np.empty(3000)
    level[0] = 50.0
    for t in range(1, 3000):  # strong pull back to 50
        level[t] = 50 + 0.7 * (level[t - 1] - 50) + rng.normal(0, 2)
    direct = price_model.simulate_direct(level, 20, kind="level")
    daily = price_model.simulate(level, 20, kind="level")
    assert np.std(direct) < 0.6 * np.std(daily)


def test_forecast_uses_the_fetched_series(monkeypatch):
    dates = business_days(1500)
    series = price_model.Series("yahoo", "BZ=F", dates, random_walk(1500))
    monkeypatch.setattr(price_model, "fetch", lambda source, symbol: series)
    result = price_model.forecast("yahoo", "BZ=F", date(2026, 10, 16), n_paths=4000)
    assert result.steps == 10
    assert "BZ=F" in result.summary() and "10 trading-day steps" in result.summary()


def question(low=80.0, high=120.0):
    return NumericQuestion(
        question_text="Where will Brent settle on 16 Oct?",
        upper_bound=high,
        lower_bound=low,
        open_upper_bound=True,
        open_lower_bound=True,
        unit_of_measure="USD",
    )


def stub_forecast(last_value, last_date, freq="business"):
    series = price_model.Series("yahoo", "BZ=F", [last_date], np.array([last_value]))
    return price_model.PriceForecast(series, date(2026, 10, 16), 10, freq, "log", 1.0, np.array([last_value]))


def test_sanity_checks_reject_stale_or_mismatched_series():
    today = date(2026, 10, 6)
    assert price_model_problem(question(), stub_forecast(101.0, date(2026, 10, 5)), today) is None
    assert "stale" in price_model_problem(question(), stub_forecast(101.0, date(2026, 9, 1)), today)
    assert "outside" in price_model_problem(question(), stub_forecast(4.4, date(2026, 10, 5)), today)
