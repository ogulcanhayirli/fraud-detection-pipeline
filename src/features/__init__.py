"""Feature computation shared by training and serving.

Every feature the model consumes is defined here and imported by both paths; no
feature logic is reimplemented anywhere else in the project.
"""

from src.features.pipeline import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    REQUIRED_INPUT_COLUMNS,
    build_features,
    compute_features_for_transaction,
)

__all__ = [
    "CATEGORICAL_FEATURES",
    "FEATURE_COLUMNS",
    "REQUIRED_INPUT_COLUMNS",
    "build_features",
    "compute_features_for_transaction",
]
