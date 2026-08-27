from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import yaml

from ..vehicle import VehicleLimits, VehicleModel, build_vehicle_model


@dataclass(frozen=True)
class TeacherVehicleConfig:
    model: VehicleModel
    ego_radius_m: float
    horizon_s: float
    dt_s: float
    mpc: dict[str, float | int]
    profile_kind: str
    sha256: str
    source: Path


def load_teacher_vehicle_config(path: str | Path) -> TeacherVehicleConfig:
    source = Path(path).resolve(); payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0": raise ValueError("teacher vehicle config schema must be 1.0")
    kind = str(payload.get("profile_kind", "physical"))
    if payload.get("parameters_confirmed") is not True:
        raise ValueError(f"{kind} vehicle parameters are not confirmed; refusing teacher generation")
    vehicle = payload.get("vehicle", {}); planning = payload.get("planning", {}); mpc = payload.get("mpc", {})
    radius = vehicle.get("ego_radius_m")
    if radius is None or float(radius) <= 0.0: raise ValueError("vehicle.ego_radius_m must be confirmed and positive")
    limits = VehicleLimits.from_mapping(vehicle.get("limits", {}))
    required_positive = (limits.forward_velocity, limits.yaw_rate, limits.acceleration,
                         limits.braking_deceleration, limits.yaw_acceleration)
    if any(value <= 0.0 for value in required_positive): raise ValueError("teacher vehicle limits are incomplete")
    model = build_vehicle_model(str(vehicle.get("variant", "unresolved")), limits, vehicle.get("geometry", {}))
    horizon, dt = float(planning.get("horizon_s", 0.0)), float(planning.get("dt_s", 0.0))
    if horizon <= 0.0 or dt <= 0.0 or not abs(round(horizon/dt)-horizon/dt) < 1e-8:
        raise ValueError("planning horizon/dt are invalid")
    return TeacherVehicleConfig(model, float(radius), horizon, dt, dict(mpc), kind,
                                sha256(source.read_bytes()).hexdigest(), source)
