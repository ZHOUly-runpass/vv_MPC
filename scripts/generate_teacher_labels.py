#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from r680_safety_planner.data import (
    TEACHER_OUTCOME_TO_CODE, load_manifest, load_training_sample, save_training_sample, write_manifest,
)
from r680_safety_planner.data.pseudo_labels import classify_solver_result
from r680_safety_planner.dcbf import CasadiDcbfSolver
from r680_safety_planner.interfaces import CandidateTrajectory, MpcRequest, MpcResult, PredictedObstacle
from r680_safety_planner.vehicle import DifferentialModel, VehicleLimits


def obstacles_from_sample(sample) -> tuple[PredictedObstacle, ...]:
    return tuple(PredictedObstacle(
        states=sample.obstacle_states[index].astype(np.float64),
        lengths=sample.obstacle_lengths[index].astype(np.float64),
        widths=sample.obstacle_widths[index].astype(np.float64),
        covariance=sample.obstacle_covariance[index].astype(np.float64),
        valid_mask=sample.obstacle_valid_mask[index],
        source="training_sample",
    ) for index in range(sample.obstacle_states.shape[0]))


def numeric_failure(candidate: CandidateTrajectory, elapsed_ms: float, error: Exception) -> MpcResult:
    return MpcResult(candidate.states.copy(), candidate.controls.copy(), False, -1.0, 1.0,
                     elapsed_ms, f"numeric_failure:{type(error).__name__}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--deadline-ms", type=float, default=80.0)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
    source_root = manifest.parent
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    limits = VehicleLimits(0.5, 0.1, 0.0, 1.0, 0.5, 0.8, 0.0, 1.0)
    model = DifferentialModel(limits)
    entries = []
    for entry in load_manifest(manifest):
        sample = load_training_sample(source_root / str(entry["path"]))
        obstacles = obstacles_from_sample(sample)
        solver = CasadiDcbfSolver(model, 0.3, hard_deadline_ms=120_000.0)
        candidates = [CandidateTrajectory(
            sample.candidate_states[index].astype(np.float64),
            sample.candidate_controls[index].astype(np.float64),
            sample.candidate_timestamps_s.astype(np.float64),
            role=f"candidate_{index}",
        ) for index in range(sample.candidate_states.shape[0])]
        if candidates:
            try:
                solver.solve(MpcRequest(sample.ego_state.astype(np.float64), candidates[-1], obstacles))
            except RuntimeError:
                pass
        results, outcomes = [], []
        for candidate in candidates:
            started = perf_counter()
            try:
                result = solver.solve(MpcRequest(sample.ego_state.astype(np.float64), candidate, obstacles))
            except Exception as error:
                result = numeric_failure(candidate, (perf_counter() - started) * 1000.0, error)
            results.append(result)
            outcomes.append(classify_solver_result(result, args.deadline_ms))
        successful = [index for index, outcome in enumerate(outcomes) if outcome == "success"]
        if successful:
            selected = min(successful, key=lambda index: (
                results[index].slack_max, float(np.mean(np.square(results[index].controls))), -results[index].h_min,
            ))
        else:
            selected = min(range(len(results)), key=lambda index: (results[index].slack_max, -results[index].h_min))
        labeled = replace(
            sample,
            teacher_outcome_codes=np.asarray([TEACHER_OUTCOME_TO_CODE[name] for name in outcomes], dtype=np.int8),
            teacher_feasible=np.asarray([result.feasible for result in results], dtype=np.bool_),
            teacher_h_min=np.asarray([result.h_min if np.isfinite(result.h_min) else 1e6 for result in results], dtype=np.float32),
            teacher_slack_max=np.asarray([result.slack_max if np.isfinite(result.slack_max) else 1e6 for result in results], dtype=np.float32),
            teacher_solve_time_ms=np.asarray([result.solve_time_ms for result in results], dtype=np.float32),
            teacher_states=np.stack([result.states for result in results]).astype(np.float32),
            teacher_controls=np.stack([result.controls for result in results]).astype(np.float32),
            teacher_selected_index=selected,
            metadata={**sample.metadata, "teacher": "casadi_dcbf_mpc", "teacher_deadline_ms": args.deadline_ms},
        )
        destination = output / "samples" / f"{sample.metadata['sample_id']}.npz"
        payload_hash = save_training_sample(destination, labeled)
        entries.append({**entry, "path": str(destination.relative_to(output)), "payload_sha256": payload_hash,
                        "teacher_outcomes": outcomes, "teacher_selected_index": selected})
    write_manifest(output / "manifest.jsonl", entries)
    print(output / "manifest.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
