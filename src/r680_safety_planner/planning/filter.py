from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..interfaces import CandidateTrajectory
from ..perception import OccupancyGrid2D
from ..vehicle import AckermannModel, MecanumModel, VehicleModel


@dataclass(frozen=True)
class CandidateCheck:
    accepted: bool
    reasons: tuple[str, ...]


class AnalyticalCandidateFilter:
    def __init__(self, model: VehicleModel) -> None:
        self.model = model

    def check(
        self,
        candidate: CandidateTrajectory,
        occupancy: OccupancyGrid2D | None = None,
    ) -> CandidateCheck:
        reasons: list[str] = []
        try:
            candidate.validate()
        except ValueError as exc:
            return CandidateCheck(False, (str(exc),))
        states = candidate.states
        limits = self.model.limits
        if isinstance(self.model, MecanumModel):
            if np.any(np.abs(states[:, 3]) > limits.forward_velocity + 1e-8):
                reasons.append("forward_velocity_limit")
            if np.any(np.abs(states[:, 4]) > limits.lateral_velocity + 1e-8):
                reasons.append("lateral_velocity_limit")
            if np.any(np.abs(states[:, 5]) > limits.yaw_rate + 1e-8):
                reasons.append("yaw_rate_limit")
        elif isinstance(self.model, AckermannModel):
            if np.any(states[:, 3] > limits.forward_velocity + 1e-8):
                reasons.append("forward_velocity_limit")
            if limits.steering_angle is not None and np.any(np.abs(candidate.controls[:, 1]) > limits.steering_angle + 1e-8):
                reasons.append("steering_limit")
        else:
            if np.any(states[:, 3] > limits.forward_velocity + 1e-8):
                reasons.append("forward_velocity_limit")
            if np.any(np.abs(states[:, 4]) > limits.yaw_rate + 1e-8):
                reasons.append("yaw_rate_limit")
        if occupancy is not None and np.any(occupancy.occupied(states[:, :2])):
            reasons.append("occupied_or_unknown_space")
        return CandidateCheck(not reasons, tuple(reasons))

