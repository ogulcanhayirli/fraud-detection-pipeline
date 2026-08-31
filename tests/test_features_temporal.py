"""Tests for row-local time features."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.temporal import age_years, day_of_week, hour_of_day


def test_hour_of_day() -> None:
    timestamps = pd.Series(pd.to_datetime(["2019-01-01 00:15", "2019-01-01 23:59"]))

    assert hour_of_day(timestamps).tolist() == [0, 23]


def test_day_of_week_uses_monday_as_zero() -> None:
    # 2019-01-07 was a Monday, 2019-01-13 the following Sunday.
    timestamps = pd.Series(pd.to_datetime(["2019-01-07", "2019-01-13"]))

    assert day_of_week(timestamps).tolist() == [0, 6]


def test_age_years() -> None:
    timestamps = pd.Series(pd.to_datetime(["2020-01-01"]))
    births = pd.Series(pd.to_datetime(["1990-01-01"]))

    assert age_years(timestamps, births).iloc[0] == pytest.approx(30.0, abs=0.02)


def test_age_is_measured_at_the_transaction_not_today() -> None:
    """The same cardholder must age between two of their own transactions.

    Computing age against the current date would make every historical row
    change on every rerun, which would silently break reproducibility.
    """
    births = pd.Series(pd.to_datetime(["1990-01-01"] * 2))
    timestamps = pd.Series(pd.to_datetime(["2019-01-01", "2020-01-01"]))

    ages = age_years(timestamps, births)

    assert ages.iloc[1] - ages.iloc[0] == pytest.approx(1.0, abs=0.01)
