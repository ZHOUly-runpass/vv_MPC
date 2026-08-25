from __future__ import annotations

from pathlib import Path
from typing import Any
import math

import numpy as np
import yaml


def load_scenario(path: str, name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    scenarios = payload.get("scenarios", {})
    if name not in scenarios:
        raise ValueError(f"unknown scenario {name!r}; choices={sorted(scenarios)}")
    return dict(payload["robot"]), dict(scenarios[name])


def dynamic_obstacles(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in scenario.get("obstacles", []) if "axis" in item]


def obstacle_catalog(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    result = [dict(item) for item in scenario.get("obstacles", [])]
    if "obstacle_prefix" in scenario:
        result.extend(
            {"name": f"{scenario['obstacle_prefix']}{index:02d}", "radius_m": scenario["obstacle_radius_m"]}
            for index in range(1, int(scenario["obstacle_count"]) + 1)
        )
    return result


def triangle_position(start: float, minimum: float, maximum: float, speed: float, time_s: float) -> float:
    width = maximum - minimum
    if width <= 0.0:
        raise ValueError("dynamic obstacle maximum must exceed minimum")
    raw = start - minimum + speed * time_s
    phase = raw % (2.0 * width)
    return minimum + (phase if phase <= width else 2.0 * width - phase)


def elevation_to_ring(xyz: np.ndarray, channels: int, minimum_deg: float, maximum_deg: float) -> np.ndarray:
    if channels < 2 or maximum_deg <= minimum_deg:
        raise ValueError("invalid LiDAR channel geometry")
    horizontal = np.hypot(xyz[:, 0], xyz[:, 1])
    elevation = np.arctan2(xyz[:, 2], horizontal)
    minimum, maximum = math.radians(minimum_deg), math.radians(maximum_deg)
    scale = (channels - 1) / (maximum - minimum)
    return np.clip(np.rint((elevation - minimum) * scale), 0, channels - 1).astype(np.uint16)


def azimuth_to_relative_time(xyz: np.ndarray, scan_period_s: float) -> np.ndarray:
    """Derive simulator firing time from its known uniform 360-degree scan."""
    if scan_period_s <= 0.0:
        raise ValueError("scan period must be positive")
    azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
    return ((azimuth + math.pi) / (2.0 * math.pi) * scan_period_s).astype(np.float32)


def rigid_transform_xyz(xyz: np.ndarray, translation: list[float], rpy: list[float]) -> np.ndarray:
    if len(translation) != 3 or len(rpy) != 3:
        raise ValueError("translation and rpy must have three values")
    roll, pitch, yaw = rpy
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    rotation = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])
    return xyz @ rotation.T + np.asarray(translation, dtype=np.float64)
