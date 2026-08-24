from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when configuration is incomplete or internally inconsistent."""


def _section(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"Missing mapping section: {key}")
    return value


@dataclass(frozen=True)
class ProjectConfig:
    raw: Mapping[str, Any]
    source: Path

    @property
    def perception_only(self) -> bool:
        return _section(self.raw, "commissioning").get("mode") == "perception_only"

    @property
    def nonzero_requested(self) -> bool:
        return bool(_section(self.raw, "commissioning").get("allow_nonzero_command", False))

    @property
    def vehicle_variant(self) -> str:
        return str(_section(self.raw, "vehicle").get("variant", "unresolved"))

    @property
    def validation_gates(self) -> Mapping[str, bool]:
        return _section(self.raw, "validation_gates")  # type: ignore[return-value]

    @property
    def failed_gates(self) -> tuple[str, ...]:
        return tuple(name for name, passed in self.validation_gates.items() if passed is not True)

    @property
    def motion_unlocked(self) -> bool:
        policy = _section(self.raw, "motion_unlock_policy")
        all_gates = not self.failed_gates if policy.get("require_all_validation_gates", True) else True
        explicit = self.nonzero_requested if policy.get(
            "require_commissioning_allow_nonzero_command", True
        ) else True
        resolved = self.vehicle_variant != "unresolved"
        return bool(all_gates and explicit and resolved and not self.perception_only)

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for key in keys:
            if not isinstance(node, Mapping) or key not in node:
                return default
            node = node[key]
        return node

    def validate(self) -> None:
        if str(self.raw.get("schema_version")) != "1.0":
            raise ConfigError("Unsupported or missing schema_version")
        if self.vehicle_variant not in {
            "unresolved",
            "differential",
            "skid_steer",
            "mecanum",
            "omni",
            "ackermann",
        }:
            raise ConfigError(f"Unsupported vehicle.variant: {self.vehicle_variant}")
        topics = _section(_section(self.raw, "ros2"), "topics")
        command = _section(topics, "command_velocity")
        if command.get("name") != "/cmd_vel":
            raise ConfigError("The guarded command output must be /cmd_vel")
        if command.get("type") != "geometry_msgs/msg/Twist":
            raise ConfigError("R680 command output must use geometry_msgs/msg/Twist")
        lidar = _section(self.raw, "lidar")
        if lidar.get("model_family") != "C16" or int(lidar.get("channels", 0)) != 16:
            raise ConfigError("This profile requires an LSLiDAR C16 16-channel sensor")
        if self.nonzero_requested and not self.motion_unlocked:
            raise ConfigError(
                "Non-zero motion was requested while commissioning gates remain closed: "
                + ", ".join(self.failed_gates)
            )


def load_project_config(path: str | Path) -> ProjectConfig:
    source = Path(path).resolve()
    with source.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, Mapping):
        raise ConfigError("Configuration root must be a mapping")
    config = ProjectConfig(raw=raw, source=source)
    config.validate()
    return config

