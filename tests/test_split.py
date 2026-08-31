"""Tests for temporal splitting.

The central guarantee is that no timestamp appears in more than one split; most
of the cases below are variations on ways that could quietly stop being true.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.data.load import LABEL_COL, TIME_COL
from src.data.split import SplitBounds, TemporalSplit, make_temporal_split, time_cutoff
from tests.conftest import make_raw_frame


@pytest.fixture
def published_files() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two frames standing in for fraudTrain.csv and fraudTest.csv."""
    train = make_raw_frame(400, start="2019-01-01", freq="h")
    test = make_raw_frame(120, start="2019-03-01", freq="h")
    return train, test


def test_no_timestamp_overlap_between_splits(
    published_files: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """The guarantee the whole module exists to provide."""
    split = make_temporal_split(*published_files)

    train_times = set(split.train[TIME_COL])
    val_times = set(split.val[TIME_COL])
    test_times = set(split.test[TIME_COL])

    assert not train_times & val_times
    assert not val_times & test_times
    assert not train_times & test_times


def test_splits_are_chronologically_ordered(
    published_files: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    split = make_temporal_split(*published_files)

    assert split.train[TIME_COL].max() < split.val[TIME_COL].min()
    assert split.val[TIME_COL].max() < split.test[TIME_COL].min()


def test_rows_are_conserved(published_files: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train_df, test_df = published_files
    split = make_temporal_split(train_df, test_df)

    assert len(split.train) + len(split.val) == len(train_df)
    assert len(split.test) == len(test_df)


def test_row_order_is_preserved(published_files: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """Nothing is shuffled: concatenating the two halves rebuilds the input order."""
    train_df, test_df = published_files
    split = make_temporal_split(train_df, test_df)

    rebuilt = pd.concat([split.train, split.val]).index.tolist()

    assert rebuilt == train_df.index.tolist()
    assert split.train.index.is_monotonic_increasing
    assert split.val.index.is_monotonic_increasing


def test_val_fraction_is_approximately_requested(
    published_files: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train_df, test_df = published_files
    split = make_temporal_split(train_df, test_df, val_fraction=0.15, snap="none")

    observed = len(split.val) / len(train_df)

    assert observed == pytest.approx(0.15, abs=0.01)


def test_duplicate_timestamps_at_cutoff_stay_in_one_split() -> None:
    """Window boundary case: tied timestamps must not straddle the cutoff."""
    train_df = make_raw_frame(100, start="2019-01-01", freq="h")
    tied_value = train_df.loc[train_df.index[85], TIME_COL]
    train_df.loc[train_df.index[80:90], TIME_COL] = tied_value
    train_df["unix_time"] = train_df[TIME_COL].astype("int64") // 10**9
    test_df = make_raw_frame(20, start="2019-06-01", freq="h")

    split = make_temporal_split(train_df, test_df, val_fraction=0.15, snap="none")

    assert not set(split.train[TIME_COL]) & set(split.val[TIME_COL])
    assert (split.val[TIME_COL] == tied_value).sum() == 10


def test_day_snap_puts_the_cutoff_at_midnight(
    published_files: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    split = make_temporal_split(*published_files, snap="day")

    assert split.val_cutoff == split.val_cutoff.normalize()


def test_snap_none_uses_an_observed_timestamp(
    published_files: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train_df, _ = published_files
    cutoff = time_cutoff(train_df[TIME_COL], 0.15, snap="none")

    assert cutoff in set(train_df[TIME_COL])


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_invalid_fraction_raises(
    fraction: float, published_files: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    train_df, _ = published_files

    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        time_cutoff(train_df[TIME_COL], fraction)


def test_single_timestamp_frame_raises() -> None:
    """A frame with no time variation cannot be split, and must say so."""
    timestamps = pd.Series([pd.Timestamp("2019-01-01")] * 10)

    with pytest.raises(ValueError, match="empty"):
        time_cutoff(timestamps, 0.15, snap="none")


def test_overlapping_published_files_raise() -> None:
    train_df = make_raw_frame(100, start="2019-01-01", freq="h")
    test_df = make_raw_frame(100, start="2019-01-02", freq="h")

    with pytest.raises(ValueError, match="overlap in time"):
        make_temporal_split(train_df, test_df)


def test_missing_column_raises() -> None:
    train_df = make_raw_frame(100).drop(columns=[LABEL_COL])
    test_df = make_raw_frame(20, start="2019-06-01")

    with pytest.raises(ValueError, match="missing required columns"):
        make_temporal_split(train_df, test_df)


def test_bounds_match_the_frames(published_files: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    split = make_temporal_split(*published_files)

    for bounds, frame in zip(split.all_bounds, (split.train, split.val, split.test), strict=True):
        assert bounds.n_rows == len(frame)
        assert bounds.start == frame[TIME_COL].min()
        assert bounds.end == frame[TIME_COL].max()
        assert bounds.n_fraud == int(frame[LABEL_COL].sum())
        assert bounds.fraud_rate == pytest.approx(bounds.n_fraud / bounds.n_rows)


def test_describe_lists_every_split(published_files: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    rendered = str(make_temporal_split(*published_files))

    for name in ("train", "val", "test"):
        assert name in rendered


def test_to_dict_is_json_serialisable(published_files: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    payload = make_temporal_split(*published_files).to_dict()

    round_tripped = json.loads(json.dumps(payload))

    assert [entry["name"] for entry in round_tripped["splits"]] == ["train", "val", "test"]


def test_temporal_split_rejects_overlapping_bounds(
    published_files: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """The invariant is enforced at construction, not left to the caller."""
    split = make_temporal_split(*published_files)
    overlapping = SplitBounds(
        name="val",
        start=split.train_bounds.start,
        end=split.val_bounds.end,
        n_rows=split.val_bounds.n_rows,
        n_fraud=split.val_bounds.n_fraud,
    )

    with pytest.raises(ValueError, match="temporal overlap"):
        TemporalSplit(
            train=split.train,
            val=split.val,
            test=split.test,
            train_bounds=split.train_bounds,
            val_bounds=overlapping,
            test_bounds=split.test_bounds,
            val_cutoff=split.val_cutoff,
            val_fraction=split.val_fraction,
            snap=split.snap,
        )
