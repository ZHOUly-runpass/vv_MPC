from hashlib import sha256

import numpy as np
import pytest

from r680_safety_planner.data import TrainingSample, load_training_sample, save_training_sample


HASH = sha256(b"fixture").hexdigest()


def sample() -> TrainingSample:
    candidates, horizon, state_size, control_size = 2, 3, 5, 2
    return TrainingSample(
        points=np.zeros((4, 6), dtype=np.float32),
        features=np.zeros((8, 4, 4), dtype=np.float32),
        route=np.zeros((5, 4), dtype=np.float32),
        ego_state=np.zeros((state_size,), dtype=np.float32),
        costmap=np.zeros((3, 4, 4), dtype=np.float32),
        obstacle_states=np.zeros((1, horizon, 6), dtype=np.float32),
        obstacle_lengths=np.ones((1, horizon), dtype=np.float32),
        obstacle_widths=np.ones((1, horizon), dtype=np.float32),
        obstacle_covariance=np.zeros((1, horizon, 2, 2), dtype=np.float32),
        obstacle_valid_mask=np.ones((1, horizon), dtype=np.bool_),
        candidate_states=np.zeros((candidates, horizon, state_size), dtype=np.float32),
        candidate_controls=np.zeros((candidates, horizon - 1, control_size), dtype=np.float32),
        candidate_timestamps_s=np.arange(horizon, dtype=np.float32),
        teacher_outcome_codes=np.array([0, 1], dtype=np.int8),
        teacher_feasible=np.array([True, False]),
        teacher_h_min=np.array([1.0, -1.0], dtype=np.float32),
        teacher_slack_max=np.array([0.0, 0.5], dtype=np.float32),
        teacher_solve_time_ms=np.array([4.0, 5.0], dtype=np.float32),
        teacher_states=np.zeros((candidates, horizon, state_size), dtype=np.float32),
        teacher_controls=np.zeros((candidates, horizon - 1, control_size), dtype=np.float32),
        teacher_selected_index=0,
        metadata={"sample_id": "fixture", "scenario": "empty", "seed": 1, "difficulty": "easy",
                  "source_sha256": HASH, "config_sha256": HASH, "checkpoint_sha256": HASH, "code_sha256": HASH},
    )


def test_training_sample_round_trip_and_hash(tmp_path) -> None:
    path = tmp_path / "sample.npz"
    digest = save_training_sample(path, sample())
    loaded = load_training_sample(path)
    assert loaded.metadata["sample_id"] == "fixture"
    assert loaded.teacher_selected_index == 0
    assert len(digest) == 64


def test_training_sample_rejects_bad_hash() -> None:
    invalid = sample().with_teacher(metadata={**sample().metadata, "code_sha256": "bad"})
    with pytest.raises(ValueError, match="code_sha256"):
        invalid.validate()
