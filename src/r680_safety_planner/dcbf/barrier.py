from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..interfaces import PredictedObstacle


def uncertainty_margin(
    covariance_xy: NDArray[np.float64],
    sigma_multiplier: float,
    relative_speed: float = 0.0,
    latency_s: float = 0.0,
    fixed_margin_m: float = 0.0,
) -> NDArray[np.float64]:
    eigenvalues = np.linalg.eigvalsh(covariance_xy)
    sigma = np.sqrt(np.maximum(eigenvalues[..., -1], 0.0))
    return fixed_margin_m + sigma_multiplier * sigma + abs(relative_speed) * latency_s


def circle_clearance(
    ego_xy: NDArray[np.float64],
    ego_radius_m: float,
    obstacle_xy: NDArray[np.float64],
    obstacle_radius_m: NDArray[np.float64],
    margin_m: NDArray[np.float64] | float = 0.0,
) -> NDArray[np.float64]:
    return np.linalg.norm(obstacle_xy - ego_xy, axis=-1) - ego_radius_m - obstacle_radius_m - margin_m


def barrier_series(
    ego_states: NDArray[np.float64],
    obstacle: PredictedObstacle,
    ego_radius_m: float,
    fixed_margin_m: float,
    sigma_multiplier: float,
    latency_s: float = 0.0,
) -> NDArray[np.float64]:
    obstacle.validate(points=ego_states.shape[0])
    obstacle_radius = 0.5 * np.hypot(obstacle.lengths, obstacle.widths)
    ego_velocity = np.linalg.norm(np.diff(ego_states[:, :2], axis=0), axis=1)
    ego_velocity = np.concatenate([ego_velocity[:1], ego_velocity])
    obstacle_velocity = np.linalg.norm(obstacle.states[:, 3:5], axis=1)
    relative_speed = float(np.max(ego_velocity + obstacle_velocity))
    margin = uncertainty_margin(
        obstacle.covariance,
        sigma_multiplier=sigma_multiplier,
        relative_speed=relative_speed,
        latency_s=latency_s,
        fixed_margin_m=fixed_margin_m,
    )
    clearance = circle_clearance(
        ego_states[:, :2], ego_radius_m, obstacle.states[:, :2], obstacle_radius, margin
    )
    return np.where(obstacle.valid_mask, clearance, np.inf)


def dcbf_residual(
    h_values: NDArray[np.float64],
    continuous_alpha: float,
    dt_s: float,
    slack: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    if h_values.ndim != 1 or h_values.size < 2:
        raise ValueError("h_values must be a vector with at least two points")
    gamma = 1.0 - np.exp(-continuous_alpha * dt_s)
    result = h_values[1:] - (1.0 - gamma) * h_values[:-1]
    if slack is not None:
        if slack.shape != result.shape:
            raise ValueError("slack shape must match D-CBF intervals")
        result = result + slack
    return result

