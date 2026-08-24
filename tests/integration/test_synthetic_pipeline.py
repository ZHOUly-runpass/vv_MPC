from pathlib import Path

from r680_safety_planner.config import load_project_config
from r680_safety_planner.pipeline import run_synthetic_smoke


def test_synthetic_pipeline_respects_motion_lock() -> None:
    config = load_project_config(Path(__file__).parents[2] / "configs" / "robot" / "r680_c16.yaml")
    result = run_synthetic_smoke(config)
    assert result["motion_allowed"] is False
    assert result["command"] == [0.0, 0.0, 0.0]
    assert "motion_locked" in result["safety_codes"]

