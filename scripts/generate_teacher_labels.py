#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from r680_safety_planner.data import (
    TEACHER_OUTCOME_TO_CODE, load_manifest, load_teacher_vehicle_config, load_training_sample,
    save_training_sample, write_manifest,
)
from r680_safety_planner.data.pseudo_labels import classify_solver_result
from r680_safety_planner.dcbf import CasadiDcbfSolver
from r680_safety_planner.interfaces import CandidateTrajectory, MpcRequest, MpcResult, PredictedObstacle
from r680_safety_planner.planning import RESAMPLING_RULE, resample_candidate_batch, resample_obstacle_batch


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
    parser.add_argument("--vehicle-config", type=Path, default=Path("configs/robot/r680_sim.yaml"))
    parser.add_argument(
        "--zero-obstacle-covariance",
        action="store_true",
        help="Zero obstacle covariance before MPC solving and persist a versioned teacher ablation.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
    source_root = manifest.parent
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    vehicle_config_path = args.vehicle_config if args.vehicle_config.is_absolute() else root/args.vehicle_config
    vehicle = load_teacher_vehicle_config(vehicle_config_path); model = vehicle.model
    entries = []
    for entry in load_manifest(manifest):
        sample = load_training_sample(source_root / str(entry["path"]))
        source_timestamps = sample.candidate_timestamps_s.copy()
        candidate_states, candidate_controls, target_timestamps = resample_candidate_batch(
            sample.candidate_states, sample.candidate_controls, source_timestamps, vehicle.dt_s, model)
        candidate_states = np.stack([model.rollout(sample.ego_state.astype(np.float64), controls.astype(np.float64), vehicle.dt_s)
                                     for controls in candidate_controls]).astype(np.float32)
        source_obstacle_covariance = (
            np.zeros_like(sample.obstacle_covariance)
            if args.zero_obstacle_covariance else sample.obstacle_covariance
        )
        obstacle_values = resample_obstacle_batch(
            sample.obstacle_states, sample.obstacle_lengths, sample.obstacle_widths, source_obstacle_covariance,
            sample.obstacle_valid_mask, source_timestamps, target_timestamps)
        sample = replace(sample, candidate_states=candidate_states, candidate_controls=candidate_controls,
                         candidate_timestamps_s=target_timestamps, obstacle_states=obstacle_values[0],
                         obstacle_lengths=obstacle_values[1], obstacle_widths=obstacle_values[2],
                         obstacle_covariance=obstacle_values[3], obstacle_valid_mask=obstacle_values[4])
        obstacles = obstacles_from_sample(sample)
        solver = CasadiDcbfSolver(
            model, vehicle.ego_radius_m, fixed_margin_m=float(vehicle.mpc.get("fixed_margin_m", 0.1)),
            sigma_multiplier=float(vehicle.mpc.get("sigma_multiplier", 3.0)),
            continuous_alpha=float(vehicle.mpc.get("continuous_alpha", 1.0)),
            maximum_slack=float(vehicle.mpc.get("maximum_slack", 1.0)), hard_deadline_ms=args.deadline_ms,
            max_iterations=int(vehicle.mpc.get("max_iterations", 100)))
        candidates = [CandidateTrajectory(
            sample.candidate_states[index].astype(np.float64),
            sample.candidate_controls[index].astype(np.float64),
            sample.candidate_timestamps_s.astype(np.float64),
            role=f"candidate_{index}",
        ) for index in range(sample.candidate_states.shape[0])]
        stop_index = min(range(len(candidates)), key=lambda index: float(np.mean(np.abs(candidates[index].states[:, 3:]))))
        stop_candidate = candidates[stop_index]
        if candidates:
            try:
                solver.solve(MpcRequest(sample.ego_state.astype(np.float64), stop_candidate, obstacles),
                             initial_guess=stop_candidate)
            except RuntimeError:
                pass
        results, outcomes, diagnostics = [], [], []
        for candidate in candidates:
            started = perf_counter()
            try:
                result = solver.solve(MpcRequest(sample.ego_state.astype(np.float64), candidate, obstacles))
                attempts = [dict(solver.last_diagnostics)]
                if "maximum_iterations" in result.status.lower():
                    retried = solver.solve(MpcRequest(sample.ego_state.astype(np.float64), candidate, obstacles),
                                           initial_guess=stop_candidate)
                    attempts.append(dict(solver.last_diagnostics)); total_ms = result.solve_time_ms+retried.solve_time_ms
                    if retried.feasible and total_ms <= args.deadline_ms:
                        result = replace(retried, solve_time_ms=total_ms,
                                         status="Solve_Succeeded_After_Safe_Stop_Retry")
                    elif total_ms > args.deadline_ms:
                        result = replace(retried, feasible=False, solve_time_ms=total_ms, status="deadline_exceeded")
                    else:
                        result = replace(retried, solve_time_ms=total_ms)
            except Exception as error:
                result = numeric_failure(candidate, (perf_counter() - started) * 1000.0, error)
                attempts = [{"return_status": result.status, "exception": type(error).__name__}]
            results.append(result)
            outcomes.append(classify_solver_result(result, args.deadline_ms))
            diagnostics.append({"candidate_index": len(results)-1, "safe_stop_candidate_index": stop_index,
                                "attempts": attempts})
        successful = [index for index, outcome in enumerate(outcomes) if outcome == "success"]
        if successful:
            selected = min(successful, key=lambda index: (
                results[index].slack_max, float(np.mean(np.square(results[index].controls))), -results[index].h_min,
            ))
        else:
            selected = min(range(len(results)), key=lambda index: (results[index].slack_max, -results[index].h_min))
        teacher_ablation = "zero_obstacle_covariance" if args.zero_obstacle_covariance else "none"
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
            metadata={**sample.metadata, "teacher": "casadi_dcbf_mpc", "teacher_deadline_ms": args.deadline_ms,
                      "teacher_vehicle_profile": vehicle.source.name, "teacher_vehicle_config_sha256": vehicle.sha256,
                      "teacher_profile_kind": vehicle.profile_kind, "teacher_dt_s": vehicle.dt_s,
                      "teacher_ablation": teacher_ablation,
                      "legacy_resampling_rule": RESAMPLING_RULE,
                      "teacher_candidate_anchor": "ego_state_rerollout",
                      "teacher_solver_statuses": [result.status for result in results],
                      "teacher_solver_diagnostics": diagnostics},
        )
        destination = output / "samples" / f"{sample.metadata['sample_id']}.npz"
        payload_hash = save_training_sample(destination, labeled)
        entries.append({**entry, "path": str(destination.relative_to(output)), "payload_sha256": payload_hash,
                        "teacher_outcomes": outcomes, "teacher_selected_index": selected,
                        "teacher_ablation": teacher_ablation})
    write_manifest(output / "manifest.jsonl", entries)
    print(output / "manifest.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
