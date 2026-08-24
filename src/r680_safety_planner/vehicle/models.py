from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from ..interfaces import ExecutionCommand


def wrap_angle(angle: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


@dataclass(frozen=True)
class VehicleLimits:
    forward_velocity: float
    reverse_velocity: float
    lateral_velocity: float
    yaw_rate: float
    acceleration: float
    braking_deceleration: float
    lateral_acceleration: float
    yaw_acceleration: float
    steering_angle: float | None = None
    steering_rate: float | None = None

    @classmethod
    def from_mapping(cls, values: Mapping[str, float | None]) -> "VehicleLimits":
        def positive(name: str, fallback: float = 0.0) -> float:
            value = values.get(name)
            return fallback if value is None else abs(float(value))

        return cls(
            forward_velocity=positive("forward_velocity_max_mps"),
            reverse_velocity=positive("reverse_velocity_max_mps"),
            lateral_velocity=positive("lateral_velocity_max_mps"),
            yaw_rate=positive("yaw_rate_max_radps"),
            acceleration=positive("acceleration_max_mps2"),
            braking_deceleration=positive("braking_deceleration_max_mps2"),
            lateral_acceleration=positive("lateral_acceleration_max_mps2"),
            yaw_acceleration=positive("yaw_acceleration_max_radps2", positive("yaw_rate_max_radps")),
            steering_angle=(
                None if values.get("steering_angle_max_rad") is None else positive("steering_angle_max_rad")
            ),
            steering_rate=(
                None if values.get("steering_rate_max_radps") is None else positive("steering_rate_max_radps")
            ),
        )


class VehicleModel(ABC):
    variant: str
    state_size: int
    control_size: int

    def __init__(self, limits: VehicleLimits) -> None:
        self.limits = limits

    @abstractmethod
    def step(self, state: NDArray[np.float64], control: NDArray[np.float64], dt: float) -> NDArray[np.float64]:
        ...

    @abstractmethod
    def command(self, state: NDArray[np.float64], control: NDArray[np.float64], timestamp_s: float) -> ExecutionCommand:
        ...

    @abstractmethod
    def stop_control(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        ...

    def rollout(
        self,
        initial_state: NDArray[np.float64],
        controls: NDArray[np.float64],
        dt: float,
    ) -> NDArray[np.float64]:
        if initial_state.shape != (self.state_size,):
            raise ValueError(f"{self.variant} state must have shape [{self.state_size}]")
        if controls.ndim != 2 or controls.shape[1] != self.control_size:
            raise ValueError(f"{self.variant} controls must have shape [N,{self.control_size}]")
        states = np.empty((controls.shape[0] + 1, self.state_size), dtype=np.float64)
        states[0] = initial_state
        for index, control in enumerate(controls):
            states[index + 1] = self.step(states[index], control, dt)
        return states


class DifferentialModel(VehicleModel):
    variant = "differential"
    state_size = 5
    control_size = 2

    def step(self, state: NDArray[np.float64], control: NDArray[np.float64], dt: float) -> NDArray[np.float64]:
        x, y, yaw, velocity, yaw_rate = state
        acceleration = float(np.clip(control[0], -self.limits.braking_deceleration, self.limits.acceleration))
        yaw_acceleration = float(np.clip(control[1], -self.limits.yaw_acceleration, self.limits.yaw_acceleration))
        velocity_next = float(np.clip(velocity + dt * acceleration, -self.limits.reverse_velocity, self.limits.forward_velocity))
        yaw_rate_next = float(np.clip(yaw_rate + dt * yaw_acceleration, -self.limits.yaw_rate, self.limits.yaw_rate))
        return np.array(
            [x + dt * velocity * np.cos(yaw), y + dt * velocity * np.sin(yaw), float(wrap_angle(yaw + dt * yaw_rate)), velocity_next, yaw_rate_next],
            dtype=np.float64,
        )

    def command(self, state: NDArray[np.float64], control: NDArray[np.float64], timestamp_s: float) -> ExecutionCommand:
        next_state = self.step(state, control, 0.05)
        return ExecutionCommand(next_state[3], 0.0, next_state[4], timestamp_s, self.variant)

    def stop_control(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array(
            [-np.sign(state[3]) * self.limits.braking_deceleration, -np.sign(state[4]) * self.limits.yaw_acceleration],
            dtype=np.float64,
        )


class SkidSteerModel(DifferentialModel):
    variant = "skid_steer"

    def __init__(self, limits: VehicleLimits, slip_yaw_gain: float = 1.0) -> None:
        super().__init__(limits)
        self.slip_yaw_gain = float(slip_yaw_gain)

    def step(self, state: NDArray[np.float64], control: NDArray[np.float64], dt: float) -> NDArray[np.float64]:
        result = super().step(state, control, dt)
        result[2] = wrap_angle(state[2] + dt * state[4] * self.slip_yaw_gain)
        return result


class MecanumModel(VehicleModel):
    variant = "mecanum"
    state_size = 6
    control_size = 3

    def step(self, state: NDArray[np.float64], control: NDArray[np.float64], dt: float) -> NDArray[np.float64]:
        x, y, yaw, vx, vy, yaw_rate = state
        ax = float(np.clip(control[0], -self.limits.braking_deceleration, self.limits.acceleration))
        ay = float(np.clip(control[1], -self.limits.lateral_acceleration, self.limits.lateral_acceleration))
        aw = float(np.clip(control[2], -self.limits.yaw_acceleration, self.limits.yaw_acceleration))
        c, s = np.cos(yaw), np.sin(yaw)
        vx_next = float(np.clip(vx + dt * ax, -self.limits.reverse_velocity, self.limits.forward_velocity))
        vy_next = float(np.clip(vy + dt * ay, -self.limits.lateral_velocity, self.limits.lateral_velocity))
        yaw_rate_next = float(np.clip(yaw_rate + dt * aw, -self.limits.yaw_rate, self.limits.yaw_rate))
        return np.array(
            [x + dt * (c * vx - s * vy), y + dt * (s * vx + c * vy), float(wrap_angle(yaw + dt * yaw_rate)), vx_next, vy_next, yaw_rate_next],
            dtype=np.float64,
        )

    def command(self, state: NDArray[np.float64], control: NDArray[np.float64], timestamp_s: float) -> ExecutionCommand:
        next_state = self.step(state, control, 0.05)
        return ExecutionCommand(next_state[3], next_state[4], next_state[5], timestamp_s, self.variant)

    def stop_control(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array(
            [-np.sign(state[3]) * self.limits.braking_deceleration, -np.sign(state[4]) * self.limits.lateral_acceleration, -np.sign(state[5]) * self.limits.yaw_acceleration],
            dtype=np.float64,
        )


class OmniModel(MecanumModel):
    variant = "omni"


class AckermannModel(VehicleModel):
    variant = "ackermann"
    state_size = 4
    control_size = 2

    def __init__(self, limits: VehicleLimits, wheelbase_m: float) -> None:
        super().__init__(limits)
        if wheelbase_m <= 0:
            raise ValueError("Ackermann wheelbase must be positive")
        if limits.steering_angle is None:
            raise ValueError("Ackermann steering limit is required")
        self.wheelbase_m = float(wheelbase_m)

    def step(self, state: NDArray[np.float64], control: NDArray[np.float64], dt: float) -> NDArray[np.float64]:
        x, y, yaw, velocity = state
        acceleration = float(np.clip(control[0], -self.limits.braking_deceleration, self.limits.acceleration))
        steering = float(np.clip(control[1], -self.limits.steering_angle, self.limits.steering_angle))
        velocity_next = float(np.clip(velocity + dt * acceleration, -self.limits.reverse_velocity, self.limits.forward_velocity))
        yaw_rate = velocity / self.wheelbase_m * np.tan(steering)
        return np.array(
            [x + dt * velocity * np.cos(yaw), y + dt * velocity * np.sin(yaw), float(wrap_angle(yaw + dt * yaw_rate)), velocity_next],
            dtype=np.float64,
        )

    def command(self, state: NDArray[np.float64], control: NDArray[np.float64], timestamp_s: float) -> ExecutionCommand:
        next_state = self.step(state, control, 0.05)
        steering = float(np.clip(control[1], -self.limits.steering_angle, self.limits.steering_angle))
        yaw_rate = next_state[3] / self.wheelbase_m * np.tan(steering)
        yaw_rate = float(np.clip(yaw_rate, -self.limits.yaw_rate, self.limits.yaw_rate))
        return ExecutionCommand(next_state[3], 0.0, yaw_rate, timestamp_s, self.variant)

    def stop_control(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([-np.sign(state[3]) * self.limits.braking_deceleration, 0.0], dtype=np.float64)


def build_vehicle_model(
    variant: str,
    limits: VehicleLimits,
    geometry: Mapping[str, float | None] | None = None,
) -> VehicleModel:
    geometry = geometry or {}
    if variant == "differential":
        return DifferentialModel(limits)
    if variant == "skid_steer":
        return SkidSteerModel(limits)
    if variant == "mecanum":
        return MecanumModel(limits)
    if variant == "omni":
        return OmniModel(limits)
    if variant == "ackermann":
        wheelbase = geometry.get("wheelbase_m")
        if wheelbase is None:
            raise ValueError("Ackermann wheelbase_m is unresolved")
        return AckermannModel(limits, float(wheelbase))
    raise ValueError(f"Vehicle variant is unresolved or unsupported: {variant}")

