#!/usr/bin/env python3
from __future__ import annotations

import json
import math

from r680_safety_planner.execution import CommandAdapter, SafetyInputs, SafetySupervisor
from r680_safety_planner.interfaces import ExecutionCommand
from r680_safety_planner.vehicle import VehicleLimits


def main() -> int:
    limits = VehicleLimits(0.5, 0.1, 0.0, 1.0, 0.5, 0.8, 0.0, 1.0)
    adapter = CommandAdapter("differential", limits, motion_unlocked=True)
    supervisor = SafetySupervisor(adapter, 0.2, 0.3, 0.15, 0.1, 0.1, 0.1)
    now = 10.0
    base = dict(now_s=now, command=ExecutionCommand(0.1, 0.0, 0.1, now, "test"),
                planner_heartbeat_s=now, point_cloud_s=now, odometry_s=now,
                imu_s=now, tf_s=now)
    cases = {
        "stale_command": {"command": ExecutionCommand(0.1, 0.0, 0.1, now - 1.0, "test")},
        "stale_heartbeat": {"planner_heartbeat_s": now - 1.0},
        "stale_cloud": {"point_cloud_s": now - 1.0},
        "stale_odometry": {"odometry_s": now - 1.0},
        "stale_imu": {"imu_s": now - 1.0},
        "stale_tf": {"tf_s": now - 1.0},
        "solver_timeout": {"solver_timed_out": True},
        "emergency_stop": {"emergency_stop_pressed": True},
        "hardware_fault": {"hardware_ready": False},
        "obstacle_emergency": {"obstacle_emergency": True},
        "nonfinite": {"command": ExecutionCommand(math.nan, 0.0, 0.0, now, "test")},
    }
    report = {}
    for name, overrides in cases.items():
        values = {**base, **overrides}
        decision = supervisor.evaluate(SafetyInputs(**values))
        stopped = not decision.allowed and decision.command.as_array().tolist() == [0.0, 0.0, 0.0]
        report[name] = {"stopped": stopped, "events": [event.code for event in decision.events]}
        if not stopped:
            raise RuntimeError(f"fault injection did not stop: {name}")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
