"""Shared fixtures.

The suite runs on synthetic frames that carry the real raw schema rather than on
the published CSVs, so a fresh clone can run `make test` without a 150MB
download and so every edge case (a null key field, duplicate timestamps at the
cutoff) can be constructed on demand instead of hunted for in real data.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from src.data.load import DATE_FORMATS, INDEX_ARTIFACT_COL, LABEL_COL, RAW_DTYPES, TIME_COL


def make_raw_frame(
    n_rows: int = 48,
    *,
    start: str = "2019-01-01",
    freq: str = "h",
    fraud_every: int = 8,
) -> pd.DataFrame:
    """Build a schema-valid transaction frame with evenly spaced timestamps."""
    timestamps = pd.date_range(start=start, periods=n_rows, freq=freq)
    epoch_seconds = timestamps.astype("int64") // 10**9
    frame = pd.DataFrame(
        {
            TIME_COL: timestamps,
            "cc_num": [f"400000000000{index % 3:04d}" for index in range(n_rows)],
            "merchant": ["fraud_Kirlin and Sons"] * n_rows,
            "category": ["grocery_pos"] * n_rows,
            "amt": [10.0 + index for index in range(n_rows)],
            "first": ["Ada"] * n_rows,
            "last": ["Lovelace"] * n_rows,
            "gender": ["F"] * n_rows,
            "street": ["1 Main St"] * n_rows,
            "city": ["Boston"] * n_rows,
            "state": ["MA"] * n_rows,
            # Leading zero is load-bearing: it is the regression case for reading
            # ZIP codes as integers.
            "zip": ["02101"] * n_rows,
            "lat": [42.36] * n_rows,
            "long": [-71.06] * n_rows,
            "merch_lat": [42.41] * n_rows,
            "merch_long": [-71.12] * n_rows,
            "city_pop": [654776] * n_rows,
            "job": ["Analyst"] * n_rows,
            "dob": pd.to_datetime(["1988-03-09"] * n_rows),
            "trans_num": [f"t{index:06d}" for index in range(n_rows)],
            "unix_time": epoch_seconds,
            LABEL_COL: [1 if index % fraud_every == 0 else 0 for index in range(n_rows)],
        }
    )
    return frame.astype(RAW_DTYPES)


def write_raw_csv(
    frame: pd.DataFrame,
    path: Path,
    *,
    with_index_artifact: bool = True,
) -> Path:
    """Write a frame in the publisher's on-disk format.

    Dates are formatted explicitly so the fixture exercises the same parsing path
    as the real file instead of whatever pandas happens to emit.
    """
    out = frame.copy()
    for column, date_format in DATE_FORMATS.items():
        if column in out.columns:
            out[column] = pd.to_datetime(out[column]).dt.strftime(date_format)
    if with_index_artifact:
        out.insert(0, INDEX_ARTIFACT_COL, range(len(out)))
    out.to_csv(path, index=False)
    return path


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    """A small, schema-valid transaction frame."""
    return make_raw_frame()


@pytest.fixture
def write_csv(tmp_path: Path) -> Callable[..., Path]:
    """Return a helper that writes a frame into the test's temporary directory."""

    def _write(
        frame: pd.DataFrame,
        name: str = "fraudTrain.csv",
        *,
        with_index_artifact: bool = True,
    ) -> Path:
        return write_raw_csv(frame, tmp_path / name, with_index_artifact=with_index_artifact)

    return _write
