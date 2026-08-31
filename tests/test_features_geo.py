"""Tests for great-circle distance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.geo import haversine_km


def _series(*values: float) -> pd.Series:
    return pd.Series(list(values), dtype="float64")


def test_known_city_distance() -> None:
    """Boston to New York City is about 306 km great-circle."""
    distance = haversine_km(
        _series(42.3601), _series(-71.0589), _series(40.7128), _series(-74.0060)
    )

    assert distance.iloc[0] == pytest.approx(306.0, abs=3.0)


def test_identical_points_are_zero() -> None:
    distance = haversine_km(_series(42.36), _series(-71.06), _series(42.36), _series(-71.06))

    assert distance.iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_distance_is_symmetric() -> None:
    forward = haversine_km(_series(42.36), _series(-71.06), _series(34.05), _series(-118.24))
    backward = haversine_km(_series(34.05), _series(-118.24), _series(42.36), _series(-71.06))

    assert forward.iloc[0] == pytest.approx(backward.iloc[0])


def test_antipodal_points_do_not_produce_nan() -> None:
    """Floating point can push the haversine term just past 1; clipping guards it."""
    distance = haversine_km(_series(0.0), _series(0.0), _series(0.0), _series(180.0))

    assert np.isfinite(distance.iloc[0])
    assert distance.iloc[0] == pytest.approx(20015.0, abs=5.0)


def test_missing_coordinates_propagate_as_nan() -> None:
    distance = haversine_km(_series(np.nan), _series(-71.06), _series(40.71), _series(-74.01))

    assert np.isnan(distance.iloc[0])


def test_result_keeps_the_input_index() -> None:
    latitudes = pd.Series([42.36, 34.05], index=[7, 9], dtype="float64")
    longitudes = pd.Series([-71.06, -118.24], index=[7, 9], dtype="float64")

    distance = haversine_km(latitudes, longitudes, latitudes, longitudes)

    assert distance.index.tolist() == [7, 9]
