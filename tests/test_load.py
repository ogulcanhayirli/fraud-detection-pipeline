"""Tests for raw loading and schema validation."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from src.data.load import (
    INDEX_ARTIFACT_COL,
    KEY_FIELDS,
    LABEL_COL,
    TIME_COL,
    SchemaError,
    load_raw,
    load_train_test,
)
from tests.conftest import make_raw_frame


def test_load_raw_applies_declared_dtypes(
    raw_frame: pd.DataFrame, write_csv: Callable[..., Path]
) -> None:
    loaded = load_raw(write_csv(raw_frame))

    assert loaded["cc_num"].dtype == "string"
    assert loaded["zip"].dtype == "string"
    assert loaded["city_pop"].dtype == "int32"
    assert loaded[LABEL_COL].dtype == "int8"
    assert loaded["category"].dtype == "category"


def test_load_raw_preserves_leading_zero_zip(
    raw_frame: pd.DataFrame, write_csv: Callable[..., Path]
) -> None:
    loaded = load_raw(write_csv(raw_frame))

    assert loaded["zip"].iloc[0] == "02101"


def test_load_raw_drops_publisher_index_column(
    raw_frame: pd.DataFrame, write_csv: Callable[..., Path]
) -> None:
    loaded = load_raw(write_csv(raw_frame, with_index_artifact=True))

    assert INDEX_ARTIFACT_COL not in loaded.columns


def test_load_raw_parses_dates(raw_frame: pd.DataFrame, write_csv: Callable[..., Path]) -> None:
    loaded = load_raw(write_csv(raw_frame))

    assert pd.api.types.is_datetime64_any_dtype(loaded[TIME_COL])
    assert pd.api.types.is_datetime64_any_dtype(loaded["dob"])
    assert loaded[TIME_COL].iloc[0] == pd.Timestamp("2019-01-01 00:00:00")


def test_load_raw_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_raw(tmp_path / "nope.csv")


def test_missing_column_raises(raw_frame: pd.DataFrame, write_csv: Callable[..., Path]) -> None:
    path = write_csv(raw_frame.drop(columns=["amt"]))

    with pytest.raises(SchemaError, match="missing expected columns"):
        load_raw(path)


def test_unexpected_column_raises(raw_frame: pd.DataFrame, write_csv: Callable[..., Path]) -> None:
    frame = raw_frame.copy()
    frame["surprise_score"] = 1.0

    with pytest.raises(SchemaError, match="unexpected columns"):
        load_raw(write_csv(frame))


@pytest.mark.parametrize("field", KEY_FIELDS)
def test_null_in_key_field_raises(
    field: str, raw_frame: pd.DataFrame, write_csv: Callable[..., Path]
) -> None:
    frame = raw_frame.copy()
    # `is_fraud` is a non-nullable int8, so blanking it fails during dtype
    # coercion rather than during validation. Both surface as SchemaError, which
    # is the contract this test pins down.
    frame[field] = frame[field].astype("object")
    frame.loc[frame.index[0], field] = None

    with pytest.raises(SchemaError):
        load_raw(write_csv(frame))


def test_invalid_label_value_raises(
    raw_frame: pd.DataFrame, write_csv: Callable[..., Path]
) -> None:
    frame = raw_frame.copy()
    frame[LABEL_COL] = frame[LABEL_COL].astype("int8")
    frame.loc[frame.index[0], LABEL_COL] = 7

    with pytest.raises(SchemaError, match="outside"):
        load_raw(write_csv(frame))


def test_unix_time_disagreeing_with_timestamp_raises(
    raw_frame: pd.DataFrame, write_csv: Callable[..., Path]
) -> None:
    frame = raw_frame.copy()
    # Reversing the epoch column mimics a timestamp parsed with the wrong
    # day/month order: the values stay plausible, the ordering does not.
    frame["unix_time"] = frame["unix_time"].to_numpy()[::-1]

    with pytest.raises(SchemaError, match="ordering"):
        load_raw(write_csv(frame))


def test_validation_can_be_skipped(raw_frame: pd.DataFrame, write_csv: Callable[..., Path]) -> None:
    frame = raw_frame.copy()
    frame["surprise_score"] = 1.0
    path = write_csv(frame)

    loaded = load_raw(path, validate=False)

    assert "surprise_score" in loaded.columns


def test_load_train_test_reads_both_files(tmp_path: Path) -> None:
    write_target = tmp_path
    from tests.conftest import write_raw_csv

    write_raw_csv(make_raw_frame(24), write_target / "fraudTrain.csv")
    write_raw_csv(
        make_raw_frame(12, start="2019-02-01"),
        write_target / "fraudTest.csv",
    )

    train, test = load_train_test(write_target)

    assert len(train) == 24
    assert len(test) == 12
    assert train[TIME_COL].max() < test[TIME_COL].min()
