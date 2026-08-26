#!/usr/bin/env python3
from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np

from r680_safety_planner.data import TrainingSample, save_training_sample, write_manifest
from r680_safety_planner.planning import CandidateGenerator
from r680_safety_planner.vehicle import DifferentialModel, VehicleLimits


HASH = sha256(b"r680-deterministic-training-fixture-v1").hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=12)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    sample_dir = output / "raw"
    limits = VehicleLimits(0.5, 0.1, 0.0, 1.0, 0.5, 0.8, 0.0, 1.0)
    model = DifferentialModel(limits)
    generator = CandidateGenerator(model, horizon_s=2.0, dt_s=0.2)
    entries = []
    for index in range(args.samples):
        rng = np.random.default_rng(1000 + index)
        initial = np.zeros(5, dtype=np.float64)
        initial[3] = 0.05 * (index % 3)
        route_xy = np.column_stack([np.linspace(0.0, 6.0, 16), np.full(16, 0.1 * ((index % 3) - 1))])
        candidates = generator.generate(initial, route_xy)
        obstacle_count = index % 4
        horizon = candidates[0].states.shape[0]
        obstacle_states = np.zeros((obstacle_count, horizon, 6), dtype=np.float32)
        for obstacle_index in range(obstacle_count):
            obstacle_states[obstacle_index, :, 0] = 1.5 + obstacle_index * 0.8
            obstacle_states[obstacle_index, :, 1] = (-1.0 if obstacle_index % 2 else 1.0) * 0.9
            obstacle_states[obstacle_index, :, 5] = 1.0
        points = rng.normal(size=(256, 6)).astype(np.float32)
        points[:, :3] *= np.array([4.0, 3.0, 0.5], dtype=np.float32)
        points[:, 4] = np.mod(np.arange(256), 16)
        metadata = {
            "sample_id": f"fixture_{index:04d}",
            "scenario": ("empty", "static_sparse", "crossing_pedestrian")[index % 3],
            "seed": 1000 + index,
            "difficulty": ("easy", "nominal", "hard")[index % 3],
            "timestamp_s": index * 0.1,
            "source_sha256": HASH,
            "config_sha256": HASH,
            "checkpoint_sha256": HASH,
            "code_sha256": HASH,
            "synthetic_fixture": True,
        }
        sample = TrainingSample(
            points=points,
            features=rng.normal(0.0, 0.1, (16, 8, 8)).astype(np.float32),
            route=np.column_stack([route_xy, np.zeros((16, 1)), np.full((16, 1), 0.25)]).astype(np.float32),
            ego_state=initial.astype(np.float32),
            costmap=np.zeros((3, 32, 32), dtype=np.float32),
            obstacle_states=obstacle_states,
            obstacle_lengths=np.full((obstacle_count, horizon), 0.5, dtype=np.float32),
            obstacle_widths=np.full((obstacle_count, horizon), 0.5, dtype=np.float32),
            obstacle_covariance=np.tile(np.eye(2, dtype=np.float32) * 0.01, (obstacle_count, horizon, 1, 1)),
            obstacle_valid_mask=np.ones((obstacle_count, horizon), dtype=np.bool_),
            candidate_states=np.stack([candidate.states for candidate in candidates]).astype(np.float32),
            candidate_controls=np.stack([candidate.controls for candidate in candidates]).astype(np.float32),
            candidate_timestamps_s=candidates[0].timestamps_s.astype(np.float32),
            teacher_outcome_codes=np.empty((0,), dtype=np.int8),
            teacher_feasible=np.empty((0,), dtype=np.bool_),
            teacher_h_min=np.empty((0,), dtype=np.float32),
            teacher_slack_max=np.empty((0,), dtype=np.float32),
            teacher_solve_time_ms=np.empty((0,), dtype=np.float32),
            teacher_states=np.empty((0, horizon, initial.size), dtype=np.float32),
            teacher_controls=np.empty((0, horizon - 1, model.control_size), dtype=np.float32),
            teacher_selected_index=-1,
            metadata=metadata,
        )
        path = sample_dir / f"{metadata['sample_id']}.npz"
        payload_hash = save_training_sample(path, sample)
        split = "test" if index % 10 == 0 else ("val" if index % 5 == 0 else "train")
        entries.append({"path": str(path.relative_to(output)), "payload_sha256": payload_hash,
                        "sample_id": metadata["sample_id"], "scenario": metadata["scenario"],
                        "seed": metadata["seed"], "difficulty": metadata["difficulty"], "split": split})
    write_manifest(output / "raw_manifest.jsonl", entries)
    print(output / "raw_manifest.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
