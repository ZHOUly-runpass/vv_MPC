#!/usr/bin/env python3
"""In-memory unlocked simulation; never edits config or publishes ROS commands."""

from __future__ import annotations

from copy import deepcopy
import json

import numpy as np

from r680_safety_planner.config import ProjectConfig, load_project_config
from r680_safety_planner.interfaces import LidarFrame
from r680_safety_planner.pipeline import SafetyPlanningPipeline
from r680_safety_planner.vehicle import DifferentialModel, VehicleLimits


def main() -> int:
    source = load_project_config("configs/robot/r680_c16.yaml")
    raw = deepcopy(source.raw)
    raw["commissioning"]["mode"] = "simulation"
    raw["commissioning"]["allow_nonzero_command"] = True
    raw["vehicle"]["variant"] = "differential"
    raw["vehicle"]["geometry"]["footprint_polygon_xy_m"] = [
        [-0.3, -0.25], [-0.3, 0.25], [0.3, 0.25], [0.3, -0.25]
    ]
    raw["mpc"]["enabled_after_vehicle_validation"] = True
    raw["mpc"]["dcbf"]["fixed_margin_m"] = 0.1
    raw["validation_gates"] = {name: True for name in raw["validation_gates"]}
    config = ProjectConfig(raw, source.source)
    config.validate()
    limits = VehicleLimits.from_mapping(raw["vehicle"]["commissioning_limits"])
    pipeline = SafetyPlanningPipeline(config, DifferentialModel(limits), ego_radius_m=0.4)
    rng = np.random.default_rng(11)
    xyz = np.column_stack([
        rng.normal(6.0, 0.1, 60), rng.normal(3.0, 0.1, 60), rng.normal(0.3, 0.05, 60)
    ])
    frame = LidarFrame(np.concatenate([xyz, np.zeros((60, 3))], axis=1), 1.0)
    result = pipeline.cycle(
        frame, np.zeros(5), 1.01,
        route_xy=np.array([[0.0, 0.0], [5.0, 0.0]], dtype=np.float64),
    )
    report = {
        "simulation_only": True, "motion_allowed": result.motion_allowed,
        "command": result.command.as_array().tolist(), "selected_role": result.selected_role,
        "safety_codes": result.safety_codes,
    }
    print(json.dumps(report, indent=2))
    if not result.motion_allowed or result.command.linear_x <= 0.0:
        raise RuntimeError("unlocked in-memory simulation did not produce forward motion")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
