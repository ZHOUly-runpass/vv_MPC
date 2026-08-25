from __future__ import annotations

import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
SIM = ROOT / "simulation" / "ros2_ws" / "src"
BRINGUP = SIM / "r680_sim_bringup"


def _scenario_module():
    path = BRINGUP / "r680_sim_bringup" / "scenario.py"
    spec = importlib.util.spec_from_file_location("r680_sim_scenario", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_all_eight_scenarios_have_parseable_worlds():
    config = yaml.safe_load((BRINGUP / "config" / "scenarios.yaml").read_text(encoding="utf-8"))
    assert set(config["scenarios"]) == {
        "empty", "static_sparse", "static_dense", "narrow_passage",
        "crossing_pedestrian", "head_on", "multi_dynamic", "local_minimum_trap",
    }
    worlds = SIM / "r680_sim_worlds" / "worlds"
    for scenario in config["scenarios"].values():
        world = worlds / scenario["world"]
        assert world.is_file()
        assert "${" not in world.read_text(encoding="utf-8")
        assert "libgazebo_ros_state.so" in world.read_text(encoding="utf-8")
        ET.parse(world)


def test_description_contract_contains_diff_drive_and_true_16_layers():
    xacro = (SIM / "r680_sim_description" / "urdf" / "r680_sim.urdf.xacro").read_text(encoding="utf-8")
    assert "libgazebo_ros_diff_drive.so" in xacro
    assert "<vertical><samples>16</samples>" in xacro
    assert "sensor_msgs/PointCloud2" in xacro
    ET.fromstring(xacro)


def test_ring_is_derived_from_sensor_elevation():
    scenario = _scenario_module()
    angles = np.deg2rad(np.array([-15.0, 0.0, 15.0]))
    xyz = np.stack([np.cos(angles), np.zeros(3), np.sin(angles)], axis=1)
    assert scenario.elevation_to_ring(xyz, 16, -15.0, 15.0).tolist() == [0, 8, 15]


def test_triangle_motion_is_bounded_and_repeatable():
    scenario = _scenario_module()
    positions = [scenario.triangle_position(-3.0, -3.0, 3.0, 0.8, t) for t in range(40)]
    assert all(-3.0 <= value <= 3.0 for value in positions)
    assert positions == [scenario.triangle_position(-3.0, -3.0, 3.0, 0.8, t) for t in range(40)]
