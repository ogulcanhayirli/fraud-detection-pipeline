"""Load and validate the raw Sparkov card-transaction files.

Validation here is strict and fails loudly, which is a deliberate trade against
convenience. A silently mistyped column - a ZIP code coerced to an integer, a
timestamp parsed with day and month transposed - still trains a model without
raising anything, and the damage surfaces weeks later as unexplained production
degradation. Load time is the cheapest place to catch it, so every assumption
this project makes about the raw schema is asserted here rather than trusted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

TIME_COL: Final[str] = "trans_date_trans_time"
LABEL_COL: Final[str] = "is_fraud"
DOB_COL: Final[str] = "dob"
UNIX_COL: Final[str] = "unix_time"

TRAIN_FILENAME: Final[str] = "fraudTrain.csv"
TEST_FILENAME: Final[str] = "fraudTest.csv"

# Per-file row counter written by the dataset publisher. Dropped on load because
# it is monotonic in transaction time: if it ever reached a model it would act as
# a proxy for the timestamp, leaking exactly the ordering the temporal split
# exists to protect.
INDEX_ARTIFACT_COL: Final[str] = "Unnamed: 0"

# Identifiers are read as strings, not numbers. `cc_num` is a 16-digit account
# number that invites meaningless arithmetic as an integer and loses its low
# digits as a float; `zip` loses its leading zero as an integer, which silently
# corrupts every New England ZIP in the file.
RAW_DTYPES: Final[dict[str, str]] = {
    "cc_num": "string",
    "merchant": "category",
    "category": "category",
    "amt": "float64",
    "first": "string",
    "last": "string",
    "gender": "category",
    "street": "string",
    "city": "string",
    "state": "category",
    "zip": "string",
    "lat": "float64",
    "long": "float64",
    "city_pop": "int32",
    "job": "category",
    "trans_num": "string",
    UNIX_COL: "int64",
    "merch_lat": "float64",
    "merch_long": "float64",
    LABEL_COL: "int8",
}

# Formats are explicit rather than inferred: inference is slow over 1.85M rows
# and can silently settle on the wrong day/month order.
DATE_FORMATS: Final[dict[str, str]] = {
    TIME_COL: "%Y-%m-%d %H:%M:%S",
    DOB_COL: "%Y-%m-%d",
}

EXPECTED_COLUMNS: Final[frozenset[str]] = frozenset(RAW_DTYPES) | frozenset(DATE_FORMATS)

# Fields the pipeline cannot recover from silently: a null here would either
# break the temporal ordering or produce an unlabelled training row.
KEY_FIELDS: Final[tuple[str, ...]] = (TIME_COL, "cc_num", "amt", "trans_num", LABEL_COL)


class SchemaError(ValueError):
    """Raised when raw data does not match the schema this project assumes."""


def load_raw(path: Path, *, validate: bool = True) -> pd.DataFrame:
    """Read one raw transaction CSV with declared dtypes and explicit date parsing.

    Args:
        path: Path to `fraudTrain.csv` or `fraudTest.csv`.
        validate: Run `validate_schema` after reading. Disable only in tests that
            deliberately construct malformed input.

    Returns:
        The transactions, with the publisher's index column removed.

    Raises:
        FileNotFoundError: If `path` does not exist.
        SchemaError: If the file cannot be read under the declared dtypes, if a
            date column does not match its declared format, or if `validate` is
            set and the schema check fails.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the Kaggle dataset into data/raw/ (see README)."
        )

    header = _read_header(path)
    keep = [column for column in header if column != INDEX_ARTIFACT_COL]
    dtypes = {column: dtype for column, dtype in RAW_DTYPES.items() if column in keep}

    # Dtype coercion failures are re-raised as SchemaError so that callers have a
    # single exception type to handle for "this file is not what we expect".
    try:
        frame = pd.read_csv(path, usecols=keep, dtype=dtypes)
        for column, date_format in DATE_FORMATS.items():
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column], format=date_format)
    except (ValueError, TypeError) as exc:
        raise SchemaError(f"{path}: could not read under the declared schema: {exc}") from exc

    if validate:
        validate_schema(frame, source=str(path))
    return frame


def load_train_test(
    data_dir: Path,
    *,
    validate: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both published files, preserving the publisher's temporal boundary.

    The two files are kept separate rather than concatenated: their boundary is
    the train/test split this project commits to, and merging them would discard
    the one piece of temporal structure the dataset hands us for free.

    Args:
        data_dir: Directory containing both CSVs, normally `data/raw`.
        validate: Passed through to `load_raw`.

    Returns:
        `(train, test)` in published order.
    """
    train = load_raw(data_dir / TRAIN_FILENAME, validate=validate)
    test = load_raw(data_dir / TEST_FILENAME, validate=validate)
    return train, test


def validate_schema(frame: pd.DataFrame, *, source: str) -> None:
    """Assert every schema assumption the rest of the pipeline relies on.

    Args:
        frame: A loaded transaction frame.
        source: Human-readable origin, used in error messages.

    Raises:
        SchemaError: On the first assumption that does not hold.
    """
    _validate_columns(frame, source=source)
    _validate_key_fields(frame, source=source)
    _validate_label(frame, source=source)
    _validate_timestamps(frame, source=source)
    _validate_time_consistency(frame, source=source)


def _read_header(path: Path) -> list[str]:
    """Read only the header row.

    Reading the header first lets a missing or renamed column surface as this
    project's own SchemaError instead of an opaque pandas dtype error.
    """
    return list(pd.read_csv(path, nrows=0).columns)


def _validate_columns(frame: pd.DataFrame, *, source: str) -> None:
    """Reject both missing and unexpected columns.

    Unexpected columns are an error, not a warning: a new column means the
    publisher changed the file, and silently ignoring it would let the schema
    drift away from what the feature code was written against.
    """
    present = frozenset(frame.columns)
    missing = sorted(EXPECTED_COLUMNS - present)
    unexpected = sorted(present - EXPECTED_COLUMNS)
    if missing:
        raise SchemaError(f"{source}: missing expected columns: {missing}")
    if unexpected:
        raise SchemaError(f"{source}: unexpected columns: {unexpected}")


def _validate_key_fields(frame: pd.DataFrame, *, source: str) -> None:
    """Reject nulls in fields the pipeline cannot impute its way out of."""
    null_counts = {
        column: int(frame[column].isna().sum())
        for column in KEY_FIELDS
        if frame[column].isna().any()
    }
    if null_counts:
        raise SchemaError(f"{source}: nulls in key fields: {null_counts}")


def _validate_label(frame: pd.DataFrame, *, source: str) -> None:
    """Reject labels outside {0, 1}."""
    observed = set(frame[LABEL_COL].unique().tolist())
    invalid = observed - {0, 1}
    if invalid:
        raise SchemaError(f"{source}: {LABEL_COL} contains values outside {{0, 1}}: {invalid}")


def _validate_timestamps(frame: pd.DataFrame, *, source: str) -> None:
    """Reject a transaction time column that is not a real datetime."""
    for column in DATE_FORMATS:
        if not pd.api.types.is_datetime64_any_dtype(frame[column]):
            raise SchemaError(
                f"{source}: {column} is {frame[column].dtype}, expected a datetime dtype"
            )


def _validate_time_consistency(frame: pd.DataFrame, *, source: str) -> None:
    """Cross-check the two time representations against each other.

    The file carries both a parsed timestamp and a Unix epoch column. Comparing
    them by *ordering* rather than by value is deliberate: the two need not share
    a timezone offset, but a timestamp parsed with the wrong day/month order
    scrambles the ordering immediately, which is the failure this catches. Ties
    are neutralised by sorting on the epoch column as a secondary key.
    """
    ordered = frame.sort_values([TIME_COL, UNIX_COL], kind="stable")[UNIX_COL]
    if not ordered.is_monotonic_increasing:
        raise SchemaError(
            f"{source}: {UNIX_COL} disagrees with {TIME_COL} on ordering, "
            "which usually means the timestamp was parsed with the wrong format"
        )
