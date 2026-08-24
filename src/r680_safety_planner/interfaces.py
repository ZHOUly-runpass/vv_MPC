from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def _finite(name: str, value: NDArray[np.floating]) -> None:
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains NaN or Inf")


@dataclass(frozen=True)
class LidarFrame:
    points: FloatArray
    timestamp_s: float
    frame_id: str = "laser"
    fields: tuple[str, ...] = ("x", "y", "z", "intensity", "ring", "time")
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.points.ndim != 2 or self.points.shape[1] < 3:
            raise ValueError("LidarFrame.points must have shape [N,D], D>=3")
        _finite("LidarFrame.points", self.points)
        if not np.isfinite(self.timestamp_s):
            raise ValueError("LidarFrame timestamp must be finite")


@dataclass(frozen=True)
class EgoState:
    state: FloatArray  # x,y,yaw plus model-specific velocities
    timestamp_s: float
    frame_id: str = "odom_combined"

    def validate(self, minimum_size: int = 4) -> None:
        if self.state.ndim != 1 or self.state.size < minimum_size:
            raise ValueError(f"EgoState.state must be a vector with at least {minimum_size} values")
        _finite("EgoState.state", self.state)


@dataclass(frozen=True)
class FrozenSceneFeatures:
    bev_feature: NDArray[np.floating]
    source_backend: str
    timestamp_s: float
    agent_states: FloatArray | None = None
    valid_mask: NDArray[np.bool_] | None = None
    quality: Mapping[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        if self.bev_feature.ndim != 4:
            raise ValueError("bev_feature must have shape [B,C,H,W]")
        _finite("bev_feature", self.bev_feature)
        if self.agent_states is not None:
            if self.agent_states.ndim != 3 or self.agent_states.shape[-1] != 7:
                raise ValueError("agent_states must have shape [B,A,7]")
            _finite("agent_states", self.agent_states)


@dataclass(frozen=True)
class PredictedObstacle:
    states: FloatArray  # [T,6]: x,y,yaw,vx,vy,confidence
    lengths: FloatArray
    widths: FloatArray
    covariance: FloatArray  # [T,2,2]
    valid_mask: NDArray[np.bool_]
    source: str = "geometric"

    def validate(self, points: int | None = None) -> None:
        if self.states.ndim != 2 or self.states.shape[1] < 5:
            raise ValueError("Obstacle states must have shape [T,>=5]")
        count = self.states.shape[0]
        if points is not None and count != points:
            raise ValueError("Obstacle prediction length does not match horizon")
        if self.lengths.shape != (count,) or self.widths.shape != (count,):
            raise ValueError("Obstacle sizes must have shape [T]")
        if self.covariance.shape != (count, 2, 2):
            raise ValueError("Obstacle covariance must have shape [T,2,2]")
        if self.valid_mask.shape != (count,):
            raise ValueError("Obstacle valid_mask must have shape [T]")
        _finite("Obstacle states", self.states)
        _finite("Obstacle covariance", self.covariance)


@dataclass(frozen=True)
class CandidateTrajectory:
    states: FloatArray
    controls: FloatArray
    timestamps_s: FloatArray
    role: str
    score: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.states.ndim != 2 or self.states.shape[0] != self.timestamps_s.size:
            raise ValueError("Candidate states and timestamps must have equal length")
        if self.controls.ndim != 2 or self.controls.shape[0] + 1 != self.states.shape[0]:
            raise ValueError("Candidate controls must contain one item per state interval")
        if np.any(np.diff(self.timestamps_s) <= 0.0):
            raise ValueError("Candidate timestamps must be strictly increasing")
        _finite("Candidate states", self.states)
        _finite("Candidate controls", self.controls)


@dataclass(frozen=True)
class MpcRequest:
    initial_state: FloatArray
    reference: CandidateTrajectory
    obstacles: tuple[PredictedObstacle, ...]

    def validate(self) -> None:
        self.reference.validate()
        if self.initial_state.shape != (self.reference.states.shape[1],):
            raise ValueError("MPC initial_state does not match candidate state dimension")
        for obstacle in self.obstacles:
            obstacle.validate(points=self.reference.states.shape[0])


@dataclass(frozen=True)
class MpcResult:
    states: FloatArray
    controls: FloatArray
    feasible: bool
    h_min: float
    slack_max: float
    solve_time_ms: float
    status: str


@dataclass(frozen=True)
class ExecutionCommand:
    linear_x: float = 0.0
    linear_y: float = 0.0
    angular_z: float = 0.0
    timestamp_s: float = 0.0
    source: str = "zero"

    def as_array(self) -> FloatArray:
        return np.array([self.linear_x, self.linear_y, self.angular_z], dtype=np.float64)

    def is_finite(self) -> bool:
        return bool(np.all(np.isfinite(self.as_array())) and np.isfinite(self.timestamp_s))


class SafetySeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    STOP = "stop"
    EMERGENCY = "emergency"


@dataclass(frozen=True)
class SafetyEvent:
    code: str
    severity: SafetySeverity
    timestamp_s: float
    detail: str = ""

