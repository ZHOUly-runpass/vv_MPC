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

    def __init__(self, model: VehicleModel, horizon_s: float = 2.0, dt_s: float = 0.1) -> None:
        self.model = model
        self.horizon_s = float(horizon_s)
        self.dt_s = float(dt_s)
        self.intervals = int(round(horizon_s / dt_s))
        if self.intervals <= 0 or not np.isclose(self.intervals * dt_s, horizon_s):
            raise ValueError("horizon_s must be a positive multiple of dt_s")

    def generate(
        self,
        initial_state: NDArray[np.float64],
        route_xy: NDArray[np.float64] | None = None,
        target_speed_mps: float | None = None,
    ) -> list[CandidateTrajectory]:
        if route_xy is not None:
            route_xy = np.asarray(route_xy, dtype=np.float64)
            if route_xy.ndim != 2 or route_xy.shape[0] < 2 or route_xy.shape[1] != 2:
                raise ValueError("route_xy must have shape [N,2], N>=2")
        target_speed = (self.model.limits.forward_velocity * 0.5
                        if target_speed_mps is None else float(target_speed_mps))
        target_speed = float(np.clip(target_speed, 0.0, self.model.limits.forward_velocity))
        timestamps = np.linspace(0.0, self.horizon_s, self.intervals + 1)
        candidates: list[CandidateTrajectory] = []
        for role in self.ROLES:
            controls = self._controls(initial_state, role, route_xy, target_speed)
            states = self.model.rollout(initial_state, controls, self.dt_s)
            candidate = CandidateTrajectory(states, controls, timestamps, role)
            candidate.validate()
            candidates.append(candidate)
        return candidates

    def _controls(
        self,
        state: NDArray[np.float64],
        role: str,
        route_xy: NDArray[np.float64] | None,
        target_speed: float,
    ) -> NDArray[np.float64]:
        controls = np.zeros((self.intervals, self.model.control_size), dtype=np.float64)
        limits = self.model.limits
        desired_speed = target_speed
        if role == "reduced_speed":
            desired_speed *= 0.5
        elif role == "controlled_stop":
            desired_speed = 0.0
        speed_error = desired_speed - float(state[3])
        controls[:, 0] = np.clip(speed_error / max(self.horizon_s, self.dt_s),
                                 -limits.braking_deceleration, limits.acceleration)
        heading_error = 0.0
        if route_xy is not None:
            delta = route_xy[-1] - state[:2]
            if np.linalg.norm(delta) > 1e-6:
                desired_yaw = np.arctan2(delta[1], delta[0])
                heading_error = float((desired_yaw - state[2] + np.pi) % (2.0 * np.pi) - np.pi)
        if isinstance(self.model, AckermannModel):
            steering = min(limits.steering_angle or 0.0, 0.2)
            controls[:, 1] = np.clip(heading_error, -steering, steering)
            if role == "left_or_counterclockwise_bias":
                controls[:, 1] = np.clip(controls[:, 1] + steering * 0.5, -steering, steering)
            elif role == "right_or_clockwise_bias":
                controls[:, 1] = np.clip(controls[:, 1] - steering * 0.5, -steering, steering)
            elif role in {"reduced_speed", "controlled_stop"}:
                pass
            elif role == "model_specific_escape_1":
                controls[:, 1] = steering * 0.5
                controls[:, 0] = -limits.braking_deceleration * 0.25
            elif role == "model_specific_escape_2":
                controls[:, 1] = -steering * 0.5
                controls[:, 0] = -limits.braking_deceleration * 0.25
            return controls

        if isinstance(self.model, MecanumModel):
            controls[:, 2] = np.clip(heading_error / max(self.horizon_s, self.dt_s),
                                     -limits.yaw_acceleration, limits.yaw_acceleration)
            if role == "left_or_counterclockwise_bias":
                controls[:, 1] = limits.lateral_acceleration * 0.5
            elif role == "right_or_clockwise_bias":
                controls[:, 1] = -limits.lateral_acceleration * 0.5
            elif role in {"reduced_speed", "controlled_stop"}:
                pass
            elif role == "model_specific_escape_1":
                controls[:, 2] = limits.yaw_acceleration * 0.5
            elif role == "model_specific_escape_2":
                controls[:, 2] = -limits.yaw_acceleration * 0.5
            return controls

        # Differential/skid-steer controls are [linear_acceleration, yaw_acceleration].
        desired_yaw_rate = np.clip(heading_error / max(self.horizon_s, self.dt_s),
                                   -limits.yaw_rate, limits.yaw_rate)
        controls[:, 1] = np.clip((desired_yaw_rate - state[4]) / max(self.horizon_s, self.dt_s),
                                 -limits.yaw_acceleration, limits.yaw_acceleration)
        if role == "left_or_counterclockwise_bias":
            controls[:, 1] = limits.yaw_acceleration * 0.5
        elif role == "right_or_clockwise_bias":
            controls[:, 1] = -limits.yaw_acceleration * 0.5
        elif role in {"reduced_speed", "controlled_stop"}:
            pass
        elif role == "model_specific_escape_1":
            controls[:, 1] = limits.yaw_acceleration
            controls[:, 0] = -limits.braking_deceleration
        elif role == "model_specific_escape_2":
            controls[:, 1] = -limits.yaw_acceleration
            controls[:, 0] = -limits.braking_deceleration
        return controls
