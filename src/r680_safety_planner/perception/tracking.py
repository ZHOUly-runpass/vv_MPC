from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..interfaces import PredictedObstacle
from .geometry import ClusterObservation


@dataclass
class Track:
    track_id: int
    state: NDArray[np.float64]  # x,y,vx,vy
    covariance: NDArray[np.float64]
    length: float
    width: float
    yaw: float
    timestamp_s: float
    missed_frames: int = 0


class ConstantVelocityTracker:
    def __init__(
        self,
        association_distance_m: float = 0.8,
        maximum_missed_frames: int = 3,
        process_variance: float = 0.05,
        measurement_variance: float = 0.02,
    ) -> None:
        self.association_distance_m = float(association_distance_m)
        self.maximum_missed_frames = int(maximum_missed_frames)
        self.process_variance = float(process_variance)
        self.measurement_variance = float(measurement_variance)
        self.tracks: dict[int, Track] = {}
        self._next_id = 1

    @staticmethod
    def _transition(dt: float) -> NDArray[np.float64]:
        return np.array(
            [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def update(self, observations: list[ClusterObservation], timestamp_s: float) -> list[Track]:
        for track in self.tracks.values():
            dt = max(0.0, timestamp_s - track.timestamp_s)
            transition = self._transition(dt)
            track.state = transition @ track.state
            track.covariance = transition @ track.covariance @ transition.T + np.eye(4) * self.process_variance * max(dt, 1e-3)
            track.timestamp_s = timestamp_s
            track.missed_frames += 1

        unmatched = set(range(len(observations)))
        pairs: list[tuple[float, int, int]] = []
        for track_id, track in self.tracks.items():
            for obs_index, observation in enumerate(observations):
                distance = float(np.linalg.norm(track.state[:2] - observation.center_xy))
                if distance <= self.association_distance_m:
                    pairs.append((distance, track_id, obs_index))
        used_tracks: set[int] = set()
        for _, track_id, obs_index in sorted(pairs):
            if track_id in used_tracks or obs_index not in unmatched:
                continue
            self._correct(self.tracks[track_id], observations[obs_index])
            used_tracks.add(track_id)
            unmatched.remove(obs_index)

        for obs_index in unmatched:
            obs = observations[obs_index]
            self.tracks[self._next_id] = Track(
                track_id=self._next_id,
                state=np.array([obs.center_xy[0], obs.center_xy[1], 0.0, 0.0]),
                covariance=np.eye(4, dtype=np.float64) * 0.1,
                length=obs.length,
                width=obs.width,
                yaw=obs.yaw,
                timestamp_s=timestamp_s,
            )
            self._next_id += 1

        self.tracks = {
            track_id: track
            for track_id, track in self.tracks.items()
            if track.missed_frames <= self.maximum_missed_frames
        }
        return list(self.tracks.values())

    def _correct(self, track: Track, observation: ClusterObservation) -> None:
        observation_matrix = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        measurement_covariance = np.eye(2) * self.measurement_variance
        residual = observation.center_xy - observation_matrix @ track.state
        innovation = observation_matrix @ track.covariance @ observation_matrix.T + measurement_covariance
        gain = track.covariance @ observation_matrix.T @ np.linalg.inv(innovation)
        track.state = track.state + gain @ residual
        track.covariance = (np.eye(4) - gain @ observation_matrix) @ track.covariance
        track.length = observation.length
        track.width = observation.width
        track.yaw = observation.yaw
        track.missed_frames = 0

    def predict(self, horizon_s: float, dt_s: float) -> tuple[PredictedObstacle, ...]:
        steps = int(round(horizon_s / dt_s)) + 1
        predictions: list[PredictedObstacle] = []
        for track in self.tracks.values():
            states = np.zeros((steps, 6), dtype=np.float64)
            covariance = np.zeros((steps, 2, 2), dtype=np.float64)
            state = track.state.copy()
            cov = track.covariance.copy()
            for index in range(steps):
                states[index] = [state[0], state[1], track.yaw, state[2], state[3], 1.0]
                covariance[index] = cov[:2, :2]
                transition = self._transition(dt_s)
                state = transition @ state
                cov = transition @ cov @ transition.T + np.eye(4) * self.process_variance * dt_s
            predictions.append(
                PredictedObstacle(
                    states=states,
                    lengths=np.full(steps, track.length, dtype=np.float64),
                    widths=np.full(steps, track.width, dtype=np.float64),
                    covariance=covariance,
                    valid_mask=np.ones(steps, dtype=bool),
                    source="geometric_tracker",
                )
            )
        return tuple(predictions)

