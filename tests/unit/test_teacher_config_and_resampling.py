from pathlib import Path

import numpy as np
import pytest

from r680_safety_planner.data import load_teacher_vehicle_config
from r680_safety_planner.planning import RESAMPLING_RULE, resample_candidate_batch, resample_obstacle_batch
from r680_safety_planner.vehicle import DifferentialModel, VehicleLimits


ROOT = Path(__file__).resolve().parents[2]


def model():
    return DifferentialModel(VehicleLimits(0.5, 0.3, 0.0, 0.8, 0.5, 0.8, 0.0, 1.0))


def test_sim_teacher_config_is_confirmed_but_physical_profile_is_rejected():
    config = load_teacher_vehicle_config(ROOT / "configs/robot/r680_sim.yaml")
    assert config.profile_kind == "simulation" and config.dt_s == pytest.approx(0.1)
    with pytest.raises(ValueError, match="not confirmed"):
        load_teacher_vehicle_config(ROOT / "configs/robot/r680_c16.yaml")


def test_legacy_candidates_are_rerolled_to_mpc_grid():
    source_t = np.arange(11, dtype=np.float32)*0.2
    controls = np.zeros((1, 10, 2), np.float32); controls[0,:,0] = np.arange(10)
    states = np.zeros((1, 11, 5), np.float32)
    new_states, new_controls, target = resample_candidate_batch(states, controls, source_t, 0.1, model())
    assert RESAMPLING_RULE.startswith("zero_order_hold")
    assert target.shape == (21,) and new_states.shape == (1,21,5) and new_controls.shape == (1,20,2)
    assert new_controls[0,:4,0].tolist() == [0.0,0.0,1.0,1.0]


def test_obstacle_interpolation_uses_conservative_validity():
    times = np.array([0.0, 0.2, 0.4]); target = np.arange(5)*0.1
    states = np.zeros((1,3,6),np.float32); states[0,:,0] = [0,2,4]
    sizes = np.ones((1,3),np.float32); covariance = np.tile(np.eye(2),(1,3,1,1)).astype(np.float32)
    output = resample_obstacle_batch(states,sizes,sizes,covariance,np.array([[True,False,True]]),times,target)
    assert output[0][0,:,0].tolist() == pytest.approx([0,1,2,3,4])
    assert not np.any(output[4])
