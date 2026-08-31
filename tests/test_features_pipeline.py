"""Tests for the shared feature pipeline.

Two properties matter more than any individual feature value here: that adding
future rows never changes a past row's features, and that the serving path
produces exactly what the training path produces.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.load import LABEL_COL
from src.features import FEATURE_COLUMNS, build_features, compute_features_for_transaction
from tests.conftest import make_raw_frame


@pytest.fixture
def transactions() -> pd.DataFrame:
    """A multi-card frame long enough for the 30-day windows to differentiate rows."""
    return make_raw_frame(240, start="2019-01-01", freq="6h")


def test_output_is_exactly_the_declared_feature_contract(transactions: pd.DataFrame) -> None:
    features = build_features(transactions)

    assert tuple(features.columns) == FEATURE_COLUMNS


def test_output_is_aligned_to_the_input(transactions: pd.DataFrame) -> None:
    features = build_features(transactions)

    assert features.index.tolist() == transactions.index.tolist()
    assert features["amt"].tolist() == transactions["amt"].tolist()


def test_label_never_reaches_the_feature_matrix(transactions: pd.DataFrame) -> None:
    features = build_features(transactions)

    assert LABEL_COL not in features.columns


def test_missing_input_column_raises(transactions: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        build_features(transactions.drop(columns=["merch_lat"]))


def test_non_unique_index_raises(transactions: pd.DataFrame) -> None:
    duplicated = transactions.copy()
    duplicated.index = pd.Index([0] * len(duplicated))

    with pytest.raises(ValueError, match="unique index"):
        build_features(duplicated)


def test_future_rows_do_not_change_past_features(transactions: pd.DataFrame) -> None:
    """The leakage test.

    Features for the first N rows must be identical whether or not the frame also
    contains everything that happened afterwards. If any window were to reach
    forwards, this is where it would show up.
    """
    prefix_length = 100
    prefix = transactions.iloc[:prefix_length]

    from_prefix = build_features(prefix)
    from_full = build_features(transactions).iloc[:prefix_length]

    pd.testing.assert_frame_equal(from_prefix, from_full)


def test_serving_path_matches_training_path(transactions: pd.DataFrame) -> None:
    """The train/serve skew test.

    Scoring the final transaction from its history alone must reproduce the row
    that batch training would have computed for it.
    """
    history = transactions.iloc[:-1]
    incoming = transactions.iloc[-1]

    served = compute_features_for_transaction(history, incoming)
    trained = build_features(transactions).iloc[-1]

    pd.testing.assert_series_equal(served, trained, check_names=False)


def test_serving_works_for_a_card_with_no_history(transactions: pd.DataFrame) -> None:
    """A card's first ever transaction is missing history features, not broken."""
    served = compute_features_for_transaction(transactions.iloc[:0], transactions.iloc[0])

    assert served["card_txn_count_24h"] == 0.0
    assert np.isnan(served["card_amt_mean_7d"])
    assert np.isnan(served["card_seconds_since_prev_txn"])
    assert served["amt"] == transactions["amt"].iloc[0]


def test_out_of_order_transaction_is_rejected(transactions: pd.DataFrame) -> None:
    """Serving a transaction older than its own history means upstream misordering."""
    history = transactions.iloc[1:]

    with pytest.raises(ValueError, match="out of order"):
        compute_features_for_transaction(history, transactions.iloc[0])


def test_zscore_is_missing_when_the_card_never_varied() -> None:
    """A card whose amounts are all identical has zero variance, not a zero z-score."""
    constant = make_raw_frame(10, start="2019-01-01", freq="h")
    constant["amt"] = 50.0
    constant["cc_num"] = "4000000000000000"

    features = build_features(constant)

    assert features["card_amt_zscore_30d"].isna().all()


def test_ratio_to_card_mean_is_computed(transactions: pd.DataFrame) -> None:
    features = build_features(transactions)
    computed = features["card_amt_ratio_to_mean_7d"].dropna()

    assert len(computed) > 0
    assert (computed > 0).all()
