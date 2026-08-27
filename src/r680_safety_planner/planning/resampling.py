from __future__ import annotations

import numpy as np

from ..vehicle import VehicleModel


RESAMPLING_RULE = "zero_order_hold_controls_rerollout_states_linear_obstacles"


def uniform_time_grid(timestamps_s: np.ndarray, target_dt_s: float) -> np.ndarray:
    source = np.asarray(timestamps_s, dtype=np.float64)
    if source.ndim != 1 or source.size < 2 or np.any(np.diff(source) <= 0.0): raise ValueError("invalid source time grid")
    if target_dt_s <= 0.0: raise ValueError("target dt must be positive")
    horizon = float(source[-1]-source[0]); intervals = int(round(horizon/target_dt_s))
    if intervals <= 0 or not np.isclose(intervals*target_dt_s, horizon, atol=1e-8):
        raise ValueError("target dt must divide the source horizon exactly")
    return source[0]+np.arange(intervals+1, dtype=np.float64)*target_dt_s


def resample_candidate_batch(states: np.ndarray, controls: np.ndarray, timestamps_s: np.ndarray,
                             target_dt_s: float, model: VehicleModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = uniform_time_grid(timestamps_s, target_dt_s); source = np.asarray(timestamps_s, dtype=np.float64)
    if np.isclose(np.median(np.diff(source)), target_dt_s): return states.copy(), controls.copy(), target.astype(np.float32)
    midpoint = target[:-1]-source[0]+0.5*target_dt_s
    indices = np.clip(np.searchsorted(source[1:]-source[0], midpoint, side="right"), 0, controls.shape[1]-1)
    new_controls = controls[:, indices].astype(np.float64)
    new_states = np.stack([model.rollout(states[index, 0].astype(np.float64), new_controls[index], target_dt_s)
                           for index in range(states.shape[0])])
    return new_states.astype(np.float32), new_controls.astype(np.float32), target.astype(np.float32)


def resample_obstacle_batch(states: np.ndarray, lengths: np.ndarray, widths: np.ndarray, covariance: np.ndarray,
                            valid_mask: np.ndarray, source_timestamps_s: np.ndarray, target_timestamps_s: np.ndarray):
    if states.shape[0] == 0:
        horizon = len(target_timestamps_s)
        return (np.empty((0,horizon,6),np.float32), np.empty((0,horizon),np.float32),
                np.empty((0,horizon),np.float32), np.empty((0,horizon,2,2),np.float32), np.empty((0,horizon),np.bool_))
    source, target = np.asarray(source_timestamps_s), np.asarray(target_timestamps_s)
    output_states = np.empty((states.shape[0], len(target), states.shape[2]), np.float32)
    output_cov = np.empty((states.shape[0], len(target), 2, 2), np.float32)
    output_lengths = np.empty((states.shape[0], len(target)), np.float32)
    output_widths = np.empty((states.shape[0], len(target)), np.float32)
    nearest = np.clip(np.searchsorted(source, target), 1, len(source)-1)
    output_valid = valid_mask[:, nearest-1] & valid_mask[:, nearest]
    for obstacle in range(states.shape[0]):
        for component in range(states.shape[2]):
            values = np.unwrap(states[obstacle,:,component]) if component == 2 else states[obstacle,:,component]
            output_states[obstacle,:,component] = np.interp(target, source, values)
        output_lengths[obstacle] = np.interp(target, source, lengths[obstacle])
        output_widths[obstacle] = np.interp(target, source, widths[obstacle])
        for row in range(2):
            for column in range(2): output_cov[obstacle,:,row,column] = np.interp(target, source, covariance[obstacle,:,row,column])
    return output_states, output_lengths, output_widths, output_cov, output_valid
