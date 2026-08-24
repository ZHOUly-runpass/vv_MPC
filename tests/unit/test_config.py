from pathlib import Path

import pytest
import yaml

from r680_safety_planner.config import ConfigError, ProjectConfig, load_project_config


CONFIG = Path(__file__).parents[2] / "configs" / "robot" / "r680_c16.yaml"


def test_commissioning_configuration_is_valid_and_locked() -> None:
    config = load_project_config(CONFIG)
    assert config.perception_only
    assert not config.motion_unlocked
    assert config.vehicle_variant == "unresolved"
    assert len(config.failed_gates) > 10


def test_nonzero_request_with_failed_gates_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["commissioning"]["allow_nonzero_command"] = True
    path = tmp_path / "unsafe.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="gates remain closed"):
        load_project_config(path)


def test_project_config_nested_get() -> None:
    config = load_project_config(CONFIG)
    assert config.get("lidar", "channels") == 16
    assert config.get("missing", default="fallback") == "fallback"

