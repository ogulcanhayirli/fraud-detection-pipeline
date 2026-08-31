"""Great-circle distance helpers.

Separated from the feature builders so the distance calculation itself can be
tested against known city-to-city distances, independently of any DataFrame
plumbing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Mean Earth radius (IUGG). The choice of radius changes distances by well under
# a percent, which is far below the resolution this feature is used at.
EARTH_RADIUS_KM: float = 6371.0088


def haversine_km(
    lat1: pd.Series,
    lon1: pd.Series,
    lat2: pd.Series,
    lon2: pd.Series,
) -> pd.Series:
    """Great-circle distance between two coordinate columns, in kilometres.

    Euclidean distance on raw latitude/longitude is rejected here because a
    degree of longitude shrinks with latitude - roughly 111 km at the equator but
    about 82 km at the mid-latitudes this dataset covers. Treating degrees as a
    flat plane would understate north-south separation relative to east-west by
    around a quarter, which matters for a feature whose whole job is to say
    whether two points are implausibly far apart.

    Args:
        lat1: Latitude of the first point, in degrees.
        lon1: Longitude of the first point, in degrees.
        lat2: Latitude of the second point, in degrees.
        lon2: Longitude of the second point, in degrees.

    Returns:
        Distance in kilometres, aligned to `lat1`'s index. Rows where any input
        is null produce NaN.
    """
    lat1_rad, lat2_rad = np.radians(lat1.to_numpy()), np.radians(lat2.to_numpy())
    delta_lat = lat2_rad - lat1_rad
    delta_lon = np.radians(lon2.to_numpy()) - np.radians(lon1.to_numpy())

    inner = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2.0) ** 2
    )
    # arcsin of a clipped value: floating point can push `inner` a hair above 1
    # for antipodal points, which would produce NaN instead of half the globe.
    distance = 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(inner, 0.0, 1.0)))
    return pd.Series(distance, index=lat1.index, name="haversine_km")
