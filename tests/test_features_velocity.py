"""Tests for causal per-card windows.

These are the tests that hold the project's central rule in place: a feature may
only see transactions strictly earlier than the one being scored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.velocity import (
    RollingSpec,
    causal_rolling_features,
    previous_transaction_features,
)

COUNT_1H = RollingSpec("1h", "count", "count_1h")
SUM_1H = RollingSpec("1h", "sum", "sum_1h")
MEAN_1H = RollingSpec("1h", "mean", "mean_1h")


def frame_of(
    cards: list[str],
    times: list[str],
    amounts: list[float],
    *,
    lats: list[float] | None = None,
    lons: list[float] | None = None,
) -> pd.DataFrame:
    """Build a minimal transaction frame for window tests."""
    size = len(cards)
    return pd.DataFrame(
        {
            "cc_num": cards,
            "trans_date_trans_time": pd.to_datetime(times),
            "amt": amounts,
            "merch_lat": lats if lats is not None else [42.0] * size,
            "merch_long": lons if lons is not None else [-71.0] * size,
        }
    )


def roll(frame: pd.DataFrame, *specs: RollingSpec) -> pd.DataFrame:
    return causal_rolling_features(
        frame,
        specs,
        group_col="cc_num",
        time_col="trans_date_trans_time",
        value_col="amt",
    )


def previous(frame: pd.DataFrame) -> pd.DataFrame:
    return previous_transaction_features(
        frame,
        group_col="cc_num",
        time_col="trans_date_trans_time",
        lat_col="merch_lat",
        lon_col="merch_long",
    )


def test_current_row_is_excluded_from_its_own_window() -> None:
    frame = frame_of(["a", "a"], ["2019-01-01 00:00", "2019-01-01 00:30"], [10.0, 20.0])

    result = roll(frame, COUNT_1H, SUM_1H)

    assert result["count_1h"].tolist() == [0.0, 1.0]
    assert result["sum_1h"].tolist() == [0.0, 10.0]


def test_window_includes_its_left_edge_and_excludes_its_right_edge() -> None:
    """The boundary case: `[t - window, t)`.

    A transaction exactly one hour earlier is inside a 1h window; a transaction
    at the scored instant is not.
    """
    frame = frame_of(
        ["a", "a", "a"],
        ["2019-01-01 00:00:00", "2019-01-01 01:00:00", "2019-01-01 01:00:01"],
        [10.0, 20.0, 30.0],
    )

    result = roll(frame, COUNT_1H)

    # Row 1 at 01:00:00 spans [00:00:00, 01:00:00): the 00:00:00 row is included.
    assert result["count_1h"].iloc[1] == 1.0
    # Row 2 at 01:00:01 spans [00:00:01, 01:00:01): 00:00:00 has fallen out,
    # 01:00:00 is now inside.
    assert result["count_1h"].iloc[2] == 1.0


def test_transactions_sharing_a_timestamp_do_not_see_each_other() -> None:
    """A card charged twice in the same second must not let either row see the other."""
    frame = frame_of(
        ["a", "a", "a"],
        ["2019-01-01 00:00", "2019-01-01 00:30", "2019-01-01 00:30"],
        [10.0, 20.0, 30.0],
    )

    result = roll(frame, COUNT_1H, SUM_1H)

    assert result["count_1h"].tolist() == [0.0, 1.0, 1.0]
    assert result["sum_1h"].tolist() == [0.0, 10.0, 10.0]


def test_cards_do_not_see_each_other() -> None:
    frame = frame_of(
        ["a", "b", "a"], ["2019-01-01 00:00"] * 2 + ["2019-01-01 00:30"], [10.0, 99.0, 20.0]
    )

    result = roll(frame, SUM_1H)

    assert result["sum_1h"].iloc[2] == 10.0


def test_results_realign_after_the_internal_sort() -> None:
    """Input order is arbitrary; output must line up with the rows as given."""
    frame = frame_of(
        ["a", "a", "a"],
        ["2019-01-01 02:00", "2019-01-01 00:00", "2019-01-01 01:30"],
        [30.0, 10.0, 20.0],
    )

    result = roll(frame, SUM_1H)

    # Row 0 (02:00) sees 01:30; row 1 (00:00) sees nothing; row 2 (01:30) sees nothing
    # because 00:00 is outside its 1h window.
    assert result["sum_1h"].tolist() == [20.0, 0.0, 0.0]
    assert result.index.tolist() == frame.index.tolist()


def test_counts_fill_zero_but_means_stay_missing() -> None:
    """An empty window means zero transactions, but an undefined average."""
    frame = frame_of(["a"], ["2019-01-01 00:00"], [10.0])

    result = roll(frame, COUNT_1H, MEAN_1H)

    assert result["count_1h"].iloc[0] == 0.0
    assert np.isnan(result["mean_1h"].iloc[0])


def test_rolling_rejects_a_non_unique_index() -> None:
    frame = frame_of(["a", "a"], ["2019-01-01 00:00", "2019-01-01 00:30"], [10.0, 20.0])
    frame.index = pd.Index([1, 1])

    with pytest.raises(ValueError, match="unique index"):
        roll(frame, COUNT_1H)


def test_previous_transaction_is_strictly_earlier() -> None:
    frame = frame_of(
        ["a", "a", "a"],
        ["2019-01-01 00:00", "2019-01-01 00:30", "2019-01-01 00:30"],
        [10.0, 20.0, 30.0],
    )

    result = previous(frame)

    # Both 00:30 rows look past each other to the 00:00 row, 1800 seconds back.
    assert np.isnan(result["card_seconds_since_prev_txn"].iloc[0])
    assert result["card_seconds_since_prev_txn"].tolist()[1:] == [1800.0, 1800.0]


def test_first_transaction_on_a_card_has_no_predecessor() -> None:
    frame = frame_of(["a", "b"], ["2019-01-01 00:00", "2019-01-01 01:00"], [10.0, 20.0])

    result = previous(frame)

    assert result["card_seconds_since_prev_txn"].isna().all()
    assert result["card_km_from_prev_txn"].isna().all()
    assert result["card_kmh_from_prev_txn"].isna().all()


def test_implied_speed_between_two_transactions() -> None:
    """Boston then New York an hour later implies roughly 306 km/h."""
    frame = frame_of(
        ["a", "a"],
        ["2019-01-01 00:00", "2019-01-01 01:00"],
        [10.0, 20.0],
        lats=[42.3601, 40.7128],
        lons=[-71.0589, -74.0060],
    )

    result = previous(frame)

    assert result["card_km_from_prev_txn"].iloc[1] == pytest.approx(306.0, abs=3.0)
    assert result["card_seconds_since_prev_txn"].iloc[1] == 3600.0
    assert result["card_kmh_from_prev_txn"].iloc[1] == pytest.approx(306.0, abs=3.0)


def test_previous_transaction_rejects_a_non_unique_index() -> None:
    frame = frame_of(["a", "a"], ["2019-01-01 00:00", "2019-01-01 00:30"], [10.0, 20.0])
    frame.index = pd.Index([1, 1])

    with pytest.raises(ValueError, match="unique index"):
        previous(frame)
