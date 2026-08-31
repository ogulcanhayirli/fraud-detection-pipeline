"""Row-local time features.

Everything here depends only on the transaction being scored, so these features
are causal by construction: there is no window to get wrong.
"""

from __future__ import annotations

import pandas as pd

DAYS_PER_YEAR: float = 365.25


def hour_of_day(timestamps: pd.Series) -> pd.Series:
    """Hour in [0, 23].

    Left as a plain integer rather than encoded as sine/cosine pairs: cyclical
    encoding exists to tell a *linear* model that 23:00 and 00:00 are adjacent,
    but a gradient-boosted tree splits on ranges and can isolate a late-night
    band directly, so the encoding would only obscure the feature.
    """
    return timestamps.dt.hour.astype("int8").rename("hour")


def day_of_week(timestamps: pd.Series) -> pd.Series:
    """Day of week, Monday=0 through Sunday=6."""
    return timestamps.dt.dayofweek.astype("int8").rename("day_of_week")


def age_years(timestamps: pd.Series, dates_of_birth: pd.Series) -> pd.Series:
    """Cardholder age at the moment of the transaction.

    Computed against the transaction timestamp rather than against today, so the
    value is stable no matter when the pipeline is re-run. Age computed against
    the current date would silently change every historical row on every rerun,
    which would make a model impossible to reproduce.
    """
    elapsed_days = (timestamps - dates_of_birth).dt.total_seconds() / 86_400.0
    return (elapsed_days / DAYS_PER_YEAR).rename("customer_age_years")
