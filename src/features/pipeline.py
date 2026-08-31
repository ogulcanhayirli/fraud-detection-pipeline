"""The single feature definition shared by training and serving.

`build_features` is the only place features are computed in this project. The
serving path does not reimplement it in a streaming form; it calls the same
function with the card's recent history plus the incoming transaction and reads
the last row. That makes train/serve skew structurally impossible rather than a
thing to be careful about: there is no second implementation to drift.

The cost of that choice is that scoring one transaction rebuilds features for
its whole history window. For a single card's 30-day history that is a few
hundred rows, which is cheap; it would not be an acceptable design if the
history were large, and the trade is revisited in docs/design-decisions.md.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from src.features.geo import haversine_km
from src.features.temporal import age_years, day_of_week, hour_of_day
from src.features.velocity import (
    RollingSpec,
    causal_rolling_features,
    previous_transaction_features,
)

CARD_COL: str = "cc_num"
TIME_COL: str = "trans_date_trans_time"
AMOUNT_COL: str = "amt"

REQUIRED_INPUT_COLUMNS: tuple[str, ...] = (
    CARD_COL,
    TIME_COL,
    AMOUNT_COL,
    "dob",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
    "city_pop",
    "category",
)

ROLLING_SPECS: tuple[RollingSpec, ...] = (
    RollingSpec("1h", "count", "card_txn_count_1h"),
    RollingSpec("24h", "count", "card_txn_count_24h"),
    RollingSpec("7D", "count", "card_txn_count_7d"),
    RollingSpec("24h", "sum", "card_amt_sum_24h"),
    RollingSpec("7D", "mean", "card_amt_mean_7d"),
    RollingSpec("30D", "mean", "card_amt_mean_30d"),
    RollingSpec("30D", "std", "card_amt_std_30d"),
)

#: The model's input contract. Ordered so that a training matrix and a serving
#: payload are always assembled the same way.
FEATURE_COLUMNS: tuple[str, ...] = (
    # Row-local
    "amt",
    "hour",
    "day_of_week",
    "customer_age_years",
    "home_to_merchant_km",
    "city_pop",
    "merchant_category",
    # Rolling, strictly earlier only
    "card_txn_count_1h",
    "card_txn_count_24h",
    "card_txn_count_7d",
    "card_amt_sum_24h",
    "card_amt_mean_7d",
    "card_amt_mean_30d",
    "card_amt_std_30d",
    "card_amt_zscore_30d",
    "card_amt_ratio_to_mean_7d",
    # Previous transaction on the same card
    "card_seconds_since_prev_txn",
    "card_km_from_prev_txn",
    "card_kmh_from_prev_txn",
)

CATEGORICAL_FEATURES: tuple[str, ...] = ("merchant_category",)


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute every model feature for every row, using only strictly earlier data.

    The function is causal for all rows simultaneously, which is what lets the
    same call serve a 1.85M-row training frame and a single-card history at
    serving time.

    Args:
        frame: Raw transactions as produced by `src.data.load`. The index must be
            unique; it is how rows are recovered after internal sorting.

    Returns:
        Exactly `FEATURE_COLUMNS`, aligned to `frame`'s index and row order. The
        label is deliberately not carried through, so it cannot reach a model by
        accident.

    Raises:
        ValueError: If a required input column is missing or the index is not
            unique.
    """
    missing = [column for column in REQUIRED_INPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"build_features is missing required columns: {missing}")
    if not frame.index.is_unique:
        raise ValueError("build_features requires a unique index; call reset_index() first")

    features = pd.DataFrame(index=frame.index)
    features["amt"] = frame[AMOUNT_COL].astype("float64")
    features["hour"] = hour_of_day(frame[TIME_COL])
    features["day_of_week"] = day_of_week(frame[TIME_COL])
    features["customer_age_years"] = age_years(frame[TIME_COL], frame["dob"])
    features["home_to_merchant_km"] = haversine_km(
        frame["lat"], frame["long"], frame["merch_lat"], frame["merch_long"]
    )
    features["city_pop"] = frame["city_pop"].astype("float64")
    features["merchant_category"] = frame["category"].astype("category")

    rolling = causal_rolling_features(
        frame,
        ROLLING_SPECS,
        group_col=CARD_COL,
        time_col=TIME_COL,
        value_col=AMOUNT_COL,
    )
    features[rolling.columns] = rolling

    previous = previous_transaction_features(
        frame,
        group_col=CARD_COL,
        time_col=TIME_COL,
        lat_col="merch_lat",
        lon_col="merch_long",
    )
    features[previous.columns] = previous

    # Ratios and z-scores are computed here rather than left to the model:
    # a tree cannot construct (amt - mean) / std from three separate columns,
    # because each split sees one feature at a time. Monotone transforms of a
    # single column (a log of the amount, say) are omitted for the mirror-image
    # reason - a tree is invariant to them, so they add width and no signal.
    features["card_amt_zscore_30d"] = _safe_divide(
        features["amt"] - features["card_amt_mean_30d"], features["card_amt_std_30d"]
    )
    features["card_amt_ratio_to_mean_7d"] = _safe_divide(
        features["amt"], features["card_amt_mean_7d"]
    )
    return features[list(FEATURE_COLUMNS)]


def compute_features_for_transaction(
    history: pd.DataFrame,
    transaction: Mapping[str, Any] | pd.Series,
) -> pd.Series:
    """Score-time entry point: features for one transaction given a card's history.

    This is the serving path, and it deliberately routes through `build_features`
    rather than reimplementing the windows incrementally. Any change to a feature
    definition therefore lands in training and serving at the same instant.

    Args:
        history: Earlier transactions, normally the same card's recent activity.
            May be empty, in which case every history-dependent feature is
            missing, exactly as it would be for a card's first ever transaction.
        transaction: The transaction to score, as a mapping or Series carrying
            `REQUIRED_INPUT_COLUMNS`.

    Returns:
        One row of `FEATURE_COLUMNS`.

    Raises:
        ValueError: If the transaction predates the history, which means events
            arrived out of order upstream and the caller's assumption that this
            is the newest transaction does not hold.
    """
    incoming = pd.DataFrame([dict(transaction)])
    if not history.empty:
        latest_known = history[TIME_COL].max()
        if incoming[TIME_COL].iloc[0] < latest_known:
            raise ValueError(
                f"transaction at {incoming[TIME_COL].iloc[0]} predates the supplied "
                f"history ending {latest_known}; events arrived out of order"
            )
        combined = pd.concat([history, incoming], ignore_index=True)
    else:
        combined = incoming

    return build_features(combined).iloc[-1]


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide, returning NaN where the denominator is zero or missing.

    A zero denominator here means a card whose historical amounts never varied,
    or that has no history at all. NaN says "undefined" and lets LightGBM route
    those rows on missingness; substituting 0 would assert the amount was exactly
    average, which is a different and false claim.
    """
    safe = denominator.where(denominator.notna() & (denominator != 0.0))
    return numerator / safe
