from __future__ import annotations

import numpy as np


def resample_route(points_xy: np.ndarray, spacing_m: float, length_m: float) -> np.ndarray:
    """Return an arc-length sampled local route; never extrapolates past input."""
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2:
        raise ValueError("route must have shape [N,2], N>=2")
    if spacing_m <= 0.0 or length_m <= 0.0 or not np.all(np.isfinite(points)):
        raise ValueError("route sampling parameters must be finite and positive")
    segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate([[True], segment > 1e-9])
    points = points[keep]
    if points.shape[0] < 2:
        raise ValueError("route has no nonzero segment")
    cumulative = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    samples = np.arange(0.0, min(length_m, cumulative[-1]) + spacing_m * 0.5, spacing_m)
    samples = np.minimum(samples, cumulative[-1])
    return np.column_stack([np.interp(samples, cumulative, points[:, axis]) for axis in range(2)])


def transform_xy(points_xy: np.ndarray, origin_xy: np.ndarray, origin_yaw: float) -> np.ndarray:
    """Transform world-frame points into the ego/reference frame."""
    points = np.asarray(points_xy, dtype=np.float64)
    shifted = points - np.asarray(origin_xy, dtype=np.float64)
    cosine, sine = np.cos(origin_yaw), np.sin(origin_yaw)
    rotation = np.array([[cosine, sine], [-sine, cosine]], dtype=np.float64)
    return shifted @ rotation.T


def occupancy_to_channels(grid: np.ndarray, unknown_value: int = -1) -> np.ndarray:
    """Encode occupied/free/unknown as three non-overlapping float channels."""
    values = np.asarray(grid)
    if values.ndim != 2:
        raise ValueError("occupancy grid must be two-dimensional")
    unknown = values == unknown_value
    occupied = values >= 50
    free = ~(unknown | occupied)
    return np.stack([occupied, free, unknown], axis=0).astype(np.float32)
