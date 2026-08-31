"""Temporal train / validation / test splitting.

Random splits are forbidden in this project. A card-fraud model is always
deployed against the future, so any evaluation that lets the model learn from
rows recorded after the transaction it scores reports a number the production
system can never reproduce. Every boundary here is therefore a point in time,
and `TemporalSplit` refuses to be constructed unless those boundaries are
strictly ordered - an invalid split cannot exist long enough to be trained on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Literal

import pandas as pd

from src.data.load import LABEL_COL, TIME_COL

Snap = Literal["none", "day"]

DEFAULT_VAL_FRACTION: Final[float] = 0.15


@dataclass(frozen=True)
class SplitBounds:
    """Time span and label balance of a single split.

    Kept separate from the frames themselves so the boundaries can be printed,
    serialised into the model card, and compared across runs without carrying
    1.85M rows around.
    """

    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    n_rows: int
    n_fraud: int

    @property
    def fraud_rate(self) -> float:
        """Positive-class share, the number threshold selection is sensitive to."""
        return self.n_fraud / self.n_rows if self.n_rows else 0.0

    @property
    def span_days(self) -> float:
        """Calendar days covered, first to last transaction."""
        return (self.end - self.start).total_seconds() / 86_400.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of this split's boundaries."""
        return {
            "name": self.name,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "n_rows": self.n_rows,
            "n_fraud": self.n_fraud,
            "fraud_rate": self.fraud_rate,
            "span_days": self.span_days,
        }


@dataclass(frozen=True, eq=False)
class TemporalSplit:
    """Three time-ordered splits plus the boundaries that produced them.

    Equality is disabled (`eq=False`) because the dataclass-generated `__eq__`
    would compare DataFrames elementwise and raise on the ambiguous truth value.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_bounds: SplitBounds
    val_bounds: SplitBounds
    test_bounds: SplitBounds
    val_cutoff: pd.Timestamp
    val_fraction: float
    snap: Snap

    def __post_init__(self) -> None:
        """Enforce the invariants that make this object trustworthy.

        Checking at construction rather than in a separate `validate()` means no
        caller can forget to run it, and nothing downstream has to re-derive
        whether the split is sound.

        Raises:
            ValueError: If any split is empty, if a bounds record disagrees with
                its frame, or if two splits overlap or touch in time.
        """
        for bounds, frame in zip(self.all_bounds, (self.train, self.val, self.test), strict=True):
            if bounds.n_rows == 0:
                raise ValueError(f"split {bounds.name!r} is empty")
            if bounds.n_rows != len(frame):
                raise ValueError(
                    f"split {bounds.name!r} bounds report {bounds.n_rows} rows "
                    f"but the frame holds {len(frame)}"
                )

        ordered = self.all_bounds
        for earlier, later in zip(ordered[:-1], ordered[1:], strict=True):
            if earlier.end >= later.start:
                raise ValueError(
                    f"temporal overlap: {earlier.name!r} ends at {earlier.end} but "
                    f"{later.name!r} starts at {later.start}; splits must be strictly ordered"
                )

    @property
    def all_bounds(self) -> tuple[SplitBounds, SplitBounds, SplitBounds]:
        """Bounds in chronological order."""
        return (self.train_bounds, self.val_bounds, self.test_bounds)

    def describe(self) -> str:
        """Render the split boundaries as an aligned table for logs and notebooks."""
        header = f"{'split':<6} {'start':<19} {'end':<19} {'rows':>10} {'fraud':>7} {'rate':>8}"
        lines = [
            f"temporal split (val_fraction={self.val_fraction:.2f}, snap={self.snap!r}, "
            f"val_cutoff={self.val_cutoff})",
            header,
            "-" * len(header),
        ]
        for bounds in self.all_bounds:
            lines.append(
                f"{bounds.name:<6} "
                f"{bounds.start:%Y-%m-%d %H:%M:%S} "
                f"{bounds.end:%Y-%m-%d %H:%M:%S} "
                f"{bounds.n_rows:>10,} "
                f"{bounds.n_fraud:>7,} "
                f"{bounds.fraud_rate:>7.3%}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of the whole split, for the model card."""
        return {
            "val_fraction": self.val_fraction,
            "snap": self.snap,
            "val_cutoff": self.val_cutoff.isoformat(),
            "splits": [bounds.to_dict() for bounds in self.all_bounds],
        }

    def __str__(self) -> str:
        """Print as the boundary table rather than as a wall of dataclass repr."""
        return self.describe()


def time_cutoff(
    timestamps: pd.Series,
    fraction: float,
    *,
    snap: Snap = "day",
) -> pd.Timestamp:
    """Return the instant at which the last `fraction` of rows begins.

    Rows are assigned by row count rather than by calendar span because
    transaction volume is not uniform over the year: taking the last 15% of the
    *time range* would let the size of the validation set - and with it the
    number of fraud cases it contains - drift with seasonality, and at a ~0.5%
    base rate that number is already small enough to be fragile.

    The cutoff is inclusive on the later side: a row is in the later split when
    its timestamp is `>= cutoff`. Rows sharing the cutoff timestamp therefore all
    land in the later split, which is what keeps the two sides from overlapping.

    Args:
        timestamps: Transaction times. Need not be sorted.
        fraction: Share of rows to place after the cutoff, strictly in (0, 1).
        snap: `"day"` rounds the cutoff down to midnight so the boundary does not
            fall mid-day, which keeps day-level reporting honest and costs only a
            fraction of a percent of split-size precision. `"none"` uses the exact
            observed timestamp.

    Returns:
        The cutoff timestamp.

    Raises:
        ValueError: If `fraction` is outside (0, 1), if `timestamps` is empty, or
            if the resulting cutoff would leave either side without rows.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be strictly between 0 and 1, got {fraction}")
    if timestamps.empty:
        raise ValueError("cannot compute a cutoff from an empty series")

    ordered = timestamps.sort_values(kind="stable").reset_index(drop=True)
    position = int(len(ordered) * (1.0 - fraction))
    cutoff = pd.Timestamp(ordered.iloc[min(position, len(ordered) - 1)])
    if snap == "day":
        cutoff = cutoff.normalize()

    if not bool((timestamps < cutoff).any()):
        raise ValueError(
            f"cutoff {cutoff} leaves the earlier split empty; the data may span "
            "too little time for the requested fraction"
        )
    if not bool((timestamps >= cutoff).any()):
        raise ValueError(f"cutoff {cutoff} leaves the later split empty")
    return cutoff


def make_temporal_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    snap: Snap = "day",
    time_col: str = TIME_COL,
    label_col: str = LABEL_COL,
) -> TemporalSplit:
    """Build the train / validation / test split from the two published files.

    The train/test boundary is the publisher's own, taken as given. The
    validation set is then carved off the end of the training period, so the
    model is always selected on data later than everything it was fitted on.

    No embargo gap is inserted between train and validation. That is a deliberate
    choice: the fraud label is known at transaction time, so there is no forward
    label leakage across the boundary, and a validation transaction whose rolling
    window reaches back into the training period is exactly what production looks
    like, where a model scores today using a customer's real history. Inserting a
    gap would make validation pessimistic relative to serving.

    Args:
        train_df: Contents of `fraudTrain.csv`.
        test_df: Contents of `fraudTest.csv`.
        val_fraction: Share of training rows to reserve for validation.
        snap: Cutoff snapping strategy, see `time_cutoff`.
        time_col: Timestamp column name.
        label_col: Binary label column name.

    Returns:
        A validated `TemporalSplit`.

    Raises:
        ValueError: If a required column is missing, if either frame is empty, or
            if the published files overlap in time.
    """
    _require_columns(train_df, (time_col, label_col), name="train_df")
    _require_columns(test_df, (time_col, label_col), name="test_df")
    if train_df.empty:
        raise ValueError("train_df is empty")
    if test_df.empty:
        raise ValueError("test_df is empty")

    train_times = train_df[time_col]
    test_times = test_df[time_col]
    if train_times.max() >= test_times.min():
        raise ValueError(
            f"the published files overlap in time: train ends at {train_times.max()} "
            f"but test starts at {test_times.min()}"
        )

    cutoff = time_cutoff(train_times, val_fraction, snap=snap)
    in_val = train_times >= cutoff

    # Boolean masking preserves the original row order, so nothing is shuffled.
    fit_frame = train_df.loc[~in_val]
    val_frame = train_df.loc[in_val]

    return TemporalSplit(
        train=fit_frame,
        val=val_frame,
        test=test_df,
        train_bounds=_measure("train", fit_frame, time_col=time_col, label_col=label_col),
        val_bounds=_measure("val", val_frame, time_col=time_col, label_col=label_col),
        test_bounds=_measure("test", test_df, time_col=time_col, label_col=label_col),
        val_cutoff=cutoff,
        val_fraction=val_fraction,
        snap=snap,
    )


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], *, name: str) -> None:
    """Fail with the caller's variable name rather than a bare KeyError."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _measure(name: str, frame: pd.DataFrame, *, time_col: str, label_col: str) -> SplitBounds:
    """Summarise one split's time span and label balance."""
    times = frame[time_col]
    return SplitBounds(
        name=name,
        start=pd.Timestamp(times.min()),
        end=pd.Timestamp(times.max()),
        n_rows=len(frame),
        n_fraud=int(frame[label_col].sum()),
    )
