"""Per-card velocity and rolling-window features.

Every function here computes, for each transaction, an aggregate over that
card's *strictly earlier* transactions. Two pandas behaviours carry that
guarantee, and both are load-bearing:

- `rolling(..., closed="left")` spans `[t - window, t)`, so the scored row is
  excluded and so is any other transaction sharing its exact timestamp.
- `merge_asof(..., allow_exact_matches=False)` looks strictly backwards, so a
  transaction never sees a same-second sibling as its "previous" transaction.

Same-timestamp transactions are the boundary case that matters: a card can be
charged twice in the same second, and a window that included those rows would
let a transaction see its own batch. Both settings are covered by dedicated
tests.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import pandas as pd

from src.features.geo import haversine_km

SECONDS_PER_HOUR: float = 3600.0


@dataclass(frozen=True)
class RollingSpec:
    """One rolling aggregate: what to compute, over how long, under what name."""

    window: str
    aggregation: str
    output: str


#: Aggregations for which an empty window means zero rather than unknown. A card
#: with no prior activity really has made zero transactions and spent zero, so
#: filling these is a statement of fact. Means and standard deviations are left
#: as NaN because they are genuinely undefined, and LightGBM handles missingness
#: natively - imputing them would invent history that does not exist.
ZERO_FILL_AGGREGATIONS: frozenset[str] = frozenset({"count", "sum"})


def causal_rolling_features(
    frame: pd.DataFrame,
    specs: tuple[RollingSpec, ...],
    *,
    group_col: str,
    time_col: str,
    value_col: str,
) -> pd.DataFrame:
    """Compute rolling aggregates over each group's strictly earlier rows.

    Specs sharing a window are evaluated in a single pass, so the cost is one
    grouped roll per distinct window rather than one per feature.

    Args:
        frame: Transactions. Must have a unique index; row identity is how
            results are mapped back after the internal sort.
        specs: Aggregates to compute.
        group_col: Entity to roll within, normally the card number.
        time_col: Timestamp column defining the window.
        value_col: Numeric column to aggregate.

    Returns:
        One column per spec, aligned to `frame`'s index and row order.

    Raises:
        ValueError: If `frame`'s index is not unique.
    """
    if not frame.index.is_unique:
        raise ValueError(
            "causal_rolling_features needs a unique index to realign results; "
            "call reset_index() on the input first"
        )

    ordered = frame[[group_col, time_col, value_col]].sort_values(
        [group_col, time_col], kind="stable"
    )
    grouped = ordered.groupby(group_col, observed=True, sort=False)

    by_window: dict[str, list[RollingSpec]] = defaultdict(list)
    for spec in specs:
        by_window[spec.window].append(spec)

    result = pd.DataFrame(index=frame.index)
    for window, window_specs in by_window.items():
        aggregations = sorted({spec.aggregation for spec in window_specs})
        rolled = grouped.rolling(window, on=time_col, closed="left")[value_col].agg(aggregations)

        # `rolling(on=...)` replaces the row index with the timestamp, so the
        # only way back to the original rows is positional. The rows come out in
        # the order they went in, which this assertion pins down rather than
        # assumes.
        if len(rolled) != len(ordered):
            raise RuntimeError(
                f"rolling produced {len(rolled)} rows for {len(ordered)} inputs; "
                "positional realignment is unsafe"
            )
        aligned = pd.DataFrame(
            rolled.to_numpy(), index=ordered.index, columns=list(aggregations)
        ).reindex(frame.index)

        for spec in window_specs:
            column = aligned[spec.aggregation]
            if spec.aggregation in ZERO_FILL_AGGREGATIONS:
                column = column.fillna(0.0)
            result[spec.output] = column
    return result


def previous_transaction_features(
    frame: pd.DataFrame,
    *,
    group_col: str,
    time_col: str,
    lat_col: str,
    lon_col: str,
) -> pd.DataFrame:
    """Compare each transaction with the card's previous, strictly earlier one.

    The implied travel speed between consecutive transactions is the point of
    this function: a card used in two places faster than a person could travel
    between them is the clearest single signal of a cloned card in this dataset.

    Because the lookup is strictly backwards, the elapsed time is always greater
    than zero, so the speed calculation needs no divide-by-zero guard - the
    strictness of the window is what makes the arithmetic safe.

    Args:
        frame: Transactions with a unique index.
        group_col: Entity to compare within, normally the card number.
        time_col: Timestamp column.
        lat_col: Latitude of the transaction location.
        lon_col: Longitude of the transaction location.

    Returns:
        Elapsed seconds, distance travelled and implied speed since the previous
        transaction, aligned to `frame`'s index. Rows with no prior transaction
        are NaN throughout.

    Raises:
        ValueError: If `frame`'s index is not unique.
    """
    if not frame.index.is_unique:
        raise ValueError(
            "previous_transaction_features needs a unique index to realign results; "
            "call reset_index() on the input first"
        )

    columns = [group_col, time_col, lat_col, lon_col]
    # merge_asof requires global sorting on the `on` key, not merely within groups.
    ordered = frame[columns].sort_values(time_col, kind="stable")
    ordered = ordered.assign(_row=ordered.index)

    merged = pd.merge_asof(
        ordered,
        ordered[columns + ["_row"]],
        on=time_col,
        by=group_col,
        direction="backward",
        allow_exact_matches=False,
        suffixes=("", "_prev"),
    )

    # merge_asof does not return the matched row's timestamp, so recover it by
    # mapping the previous row's identifier back through the source frame.
    previous_times = merged["_row_prev"].map(frame[time_col])
    elapsed_seconds = (merged[time_col] - previous_times).dt.total_seconds()

    distance_km = haversine_km(
        merged[lat_col],
        merged[lon_col],
        merged[f"{lat_col}_prev"],
        merged[f"{lon_col}_prev"],
    )

    result = pd.DataFrame(
        {
            "card_seconds_since_prev_txn": elapsed_seconds.to_numpy(),
            "card_km_from_prev_txn": distance_km.to_numpy(),
        },
        index=pd.Index(merged["_row"]),
    )
    result["card_kmh_from_prev_txn"] = result["card_km_from_prev_txn"] / (
        result["card_seconds_since_prev_txn"] / SECONDS_PER_HOUR
    )
    return result.reindex(frame.index)
