from __future__ import annotations

from dataclasses import dataclass

from ..interfaces import ExecutionCommand, SafetyEvent, SafetySeverity
from .command_adapter import CommandAdapter


@dataclass(frozen=True)
class SafetyInputs:
    now_s: float
    command: ExecutionCommand
    planner_heartbeat_s: float
    point_cloud_s: float
    odometry_s: float
    imu_s: float
    tf_s: float
    solver_timed_out: bool = False
    emergency_stop_pressed: bool = False
    hardware_ready: bool = True
    obstacle_emergency: bool = False
    route_ready: bool = True


@dataclass(frozen=True)
class SupervisorDecision:
    command: ExecutionCommand
    allowed: bool
    events: tuple[SafetyEvent, ...]


class SafetySupervisor:
    def __init__(
        self,
        adapter: CommandAdapter,
        command_timeout_s: float,
        heartbeat_timeout_s: float,
        point_cloud_timeout_s: float,
        odometry_timeout_s: float,
        imu_timeout_s: float,
        tf_timeout_s: float,
    ) -> None:
        self.adapter = adapter
        self.thresholds = {
            "stale_command": float(command_timeout_s),
            "stale_planner_heartbeat": float(heartbeat_timeout_s),
            "stale_point_cloud": float(point_cloud_timeout_s),
            "stale_odometry": float(odometry_timeout_s),
            "stale_imu": float(imu_timeout_s),
            "stale_tf": float(tf_timeout_s),
        }

    def evaluate(self, inputs: SafetyInputs) -> SupervisorDecision:
        failures: list[tuple[str, str]] = []
        timestamps = {
            "stale_command": inputs.command.timestamp_s,
            "stale_planner_heartbeat": inputs.planner_heartbeat_s,
            "stale_point_cloud": inputs.point_cloud_s,
            "stale_odometry": inputs.odometry_s,
            "stale_imu": inputs.imu_s,
            "stale_tf": inputs.tf_s,
        }
        for code, timestamp in timestamps.items():
            age = inputs.now_s - timestamp
            if age < -1e-6 or age > self.thresholds[code]:
                failures.append((code, f"age={age:.3f}s limit={self.thresholds[code]:.3f}s"))
        if not inputs.command.is_finite():
            failures.append(("nonfinite_command", "command contains NaN or Inf"))
        if inputs.solver_timed_out:
            failures.append(("solver_timeout", "solver exceeded hard deadline"))
        if inputs.emergency_stop_pressed:
            failures.append(("physical_emergency_stop", "emergency stop is pressed"))
        if not inputs.hardware_ready:
            failures.append(("hardware_not_ready", "controller enable/battery/diagnostics not ready"))
        if inputs.obstacle_emergency:
            failures.append(("obstacle_emergency", "obstacle is inside emergency stopping boundary"))
        if not inputs.route_ready:
            failures.append(("route_missing", "no route is available in the odometry frame"))
        if not self.adapter.motion_unlocked:
            failures.append(("motion_locked", "commissioning gates are not complete"))

        events = tuple(
            SafetyEvent(code, SafetySeverity.EMERGENCY if "emergency" in code else SafetySeverity.STOP, inputs.now_s, detail)
            for code, detail in failures
        )
        if events:
            return SupervisorDecision(CommandAdapter.zero(inputs.now_s, "watchdog_stop"), False, events)
        return SupervisorDecision(self.adapter.sanitize(inputs.command, inputs.now_s), True, ())
