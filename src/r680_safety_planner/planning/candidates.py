from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..interfaces import CandidateTrajectory
from ..vehicle import AckermannModel, MecanumModel, VehicleModel


class CandidateGenerator:
    """Deterministic seed candidates used before a learned head is available."""

    ROLES = (
        "nominal_route_following",
        "left_or_counterclockwise_bias",
        "right_or_clockwise_bias",
        "reduced_speed",
        "controlled_stop",
        "model_specific_escape_1",
        "model_specific_escape_2",
    )

    def __init__(self, model: VehicleModel, horizon_s: float = 2.0, dt_s: float = 0.2) -> None:
        self.model = model
        self.horizon_s = float(horizon_s)
        self.dt_s = float(dt_s)
        self.intervals = int(round(horizon_s / dt_s))
        if self.intervals <= 0 or not np.isclose(self.intervals * dt_s, horizon_s):
            raise ValueError("horizon_s must be a positive multiple of dt_s")

    def generate(self, initial_state: NDArray[np.float64]) -> list[CandidateTrajectory]:
        timestamps = np.linspace(0.0, self.horizon_s, self.intervals + 1)
        candidates: list[CandidateTrajectory] = []
        for role in self.ROLES:
            controls = self._controls(initial_state, role)
            states = self.model.rollout(initial_state, controls, self.dt_s)
            candidate = CandidateTrajectory(states, controls, timestamps, role)
            candidate.validate()
            candidates.append(candidate)
        return candidates

    def _controls(self, state: NDArray[np.float64], role: str) -> NDArray[np.float64]:
        controls = np.zeros((self.intervals, self.model.control_size), dtype=np.float64)
        limits = self.model.limits
        if isinstance(self.model, AckermannModel):
            steering = min(limits.steering_angle or 0.0, 0.2)
            if role == "left_or_counterclockwise_bias":
                controls[:, 1] = steering
            elif role == "right_or_clockwise_bias":
                controls[:, 1] = -steering
            elif role in {"reduced_speed", "controlled_stop"}:
                controls[:, 0] = -limits.braking_deceleration * (0.5 if role == "reduced_speed" else 1.0)
            elif role == "model_specific_escape_1":
                controls[:, 1] = steering * 0.5
                controls[:, 0] = -limits.braking_deceleration * 0.25
            elif role == "model_specific_escape_2":
                controls[:, 1] = -steering * 0.5
                controls[:, 0] = -limits.braking_deceleration * 0.25
            return controls

        if isinstance(self.model, MecanumModel):
            if role == "left_or_counterclockwise_bias":
                controls[:, 1] = limits.lateral_acceleration * 0.5
            elif role == "right_or_clockwise_bias":
                controls[:, 1] = -limits.lateral_acceleration * 0.5
            elif role in {"reduced_speed", "controlled_stop"}:
                controls[:, 0] = -limits.braking_deceleration * (0.5 if role == "reduced_speed" else 1.0)
            elif role == "model_specific_escape_1":
                controls[:, 2] = limits.yaw_acceleration * 0.5
            elif role == "model_specific_escape_2":
                controls[:, 2] = -limits.yaw_acceleration * 0.5
            return controls

        # Differential/skid-steer controls are [linear_acceleration, yaw_acceleration].
        if role == "left_or_counterclockwise_bias":
            controls[:, 1] = limits.yaw_acceleration * 0.5
        elif role == "right_or_clockwise_bias":
            controls[:, 1] = -limits.yaw_acceleration * 0.5
        elif role in {"reduced_speed", "controlled_stop"}:
            controls[:, 0] = -limits.braking_deceleration * (0.5 if role == "reduced_speed" else 1.0)
        elif role == "model_specific_escape_1":
            controls[:, 1] = limits.yaw_acceleration
            controls[:, 0] = -limits.braking_deceleration
        elif role == "model_specific_escape_2":
            controls[:, 1] = -limits.yaw_acceleration
            controls[:, 0] = -limits.braking_deceleration
        return controls

