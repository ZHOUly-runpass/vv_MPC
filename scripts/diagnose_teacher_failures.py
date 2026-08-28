#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from r680_safety_planner.data import load_manifest, load_teacher_vehicle_config, load_training_sample
from r680_safety_planner.dcbf import CasadiDcbfSolver
from r680_safety_planner.interfaces import CandidateTrajectory, MpcRequest, PredictedObstacle


def obstacles_from_sample(sample) -> tuple[PredictedObstacle, ...]:
    return tuple(PredictedObstacle(sample.obstacle_states[index].astype(np.float64),
        sample.obstacle_lengths[index].astype(np.float64), sample.obstacle_widths[index].astype(np.float64),
        sample.obstacle_covariance[index].astype(np.float64), sample.obstacle_valid_mask[index], "training_sample")
        for index in range(sample.obstacle_states.shape[0]))


def constraint_diagnostics(solver, request, active) -> dict[str, float | int]:
    reference = request.reference; dt = float(np.median(np.diff(reference.timestamps_s)))
    gamma = 1.0-np.exp(-solver.continuous_alpha*dt); required = []; h_values = []
    for obstacle in active:
        covariance_radius = solver.sigma_multiplier*np.sqrt(np.maximum(0.0, np.linalg.eigvalsh(obstacle.covariance)[:, -1]))
        safe = solver.ego_radius_m+0.5*np.hypot(obstacle.lengths, obstacle.widths)+solver.fixed_margin_m+covariance_radius
        h = np.sum((reference.states[:, :2]-obstacle.states[:, :2])**2, axis=1)-safe**2
        h_values.append(h); required.extend((-h[:-1]).tolist()); required.extend((-(h[1:]-(1.0-gamma)*h[:-1])).tolist())
    rollout = solver.model.rollout(request.initial_state, reference.controls, dt)
    return {"active_obstacles": len(active),
            "initial_h_squared_min": min((float(values[0]) for values in h_values), default=float("inf")),
            "reference_h_squared_min": min((float(np.min(values)) for values in h_values), default=float("inf")),
            "required_slack_squared": max(0.0, max(required, default=0.0)),
            "initial_reference_error": float(np.linalg.norm(reference.states[0]-request.initial_state)),
            "rollout_defect_max": float(np.max(np.abs(rollout-reference.states)))}


def summary(values: list[float]) -> dict[str, float | None]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    return {"median": None if not finite.size else float(np.median(finite)),
            "maximum": None if not finite.size else float(np.max(finite))}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--vehicle-config", type=Path, default=Path("configs/robot/r680_sim.yaml"))
    parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]; manifest = args.manifest.resolve()
    config = args.vehicle_config if args.vehicle_config.is_absolute() else root/args.vehicle_config
    vehicle = load_teacher_vehicle_config(config); solver = CasadiDcbfSolver(vehicle.model, vehicle.ego_radius_m,
        fixed_margin_m=float(vehicle.mpc.get("fixed_margin_m", .1)), sigma_multiplier=float(vehicle.mpc.get("sigma_multiplier", 3)),
        continuous_alpha=float(vehicle.mpc.get("continuous_alpha", 1)), maximum_slack=float(vehicle.mpc.get("maximum_slack", 1)),
        hard_deadline_ms=float(vehicle.mpc.get("hard_deadline_ms", 80)), max_iterations=int(vehicle.mpc.get("max_iterations", 100)))
    failures = []; group_counts = defaultdict(Counter); group_metrics = defaultdict(lambda: defaultdict(list))
    for entry in load_manifest(manifest):
        sample = load_training_sample(manifest.parent/str(entry["path"])); obstacles = obstacles_from_sample(sample)
        statuses = sample.metadata.get("teacher_solver_statuses", ["unavailable"]*sample.candidate_states.shape[0])
        group = f"{sample.metadata.get('scenario')}/{sample.metadata.get('difficulty')}"
        for index, status in enumerate(statuses):
            candidate = CandidateTrajectory(sample.candidate_states[index].astype(np.float64),
                sample.candidate_controls[index].astype(np.float64), sample.candidate_timestamps_s.astype(np.float64), f"candidate_{index}")
            request = MpcRequest(sample.ego_state.astype(np.float64), candidate, obstacles)
            active = solver.select_reachable_obstacles(request); metrics = constraint_diagnostics(solver, request, active)
            outcome = "success" if status == "Solve_Succeeded" else str(status)
            group_counts[group][outcome] += 1
            for name, value in metrics.items(): group_metrics[group][name].append(float(value))
            if status != "Solve_Succeeded":
                failures.append({"sample_id": sample.metadata["sample_id"], "group": group, "candidate_index": index,
                    "solver_status": status, "total_obstacles": len(obstacles), **metrics,
                    "solve_time_ms": float(sample.teacher_solve_time_ms[index]),
                    "reported_h_min_m": float(sample.teacher_h_min[index])})
    report = {"schema_version": "1.0", "manifest": str(manifest), "failure_count": len(failures),
        "maximum_slack_squared": solver.maximum_slack, "groups": {group: {"statuses": dict(group_counts[group]),
            "metrics": {name: summary(values) for name, values in group_metrics[group].items()}}
            for group in sorted(group_counts)}, "failures": failures}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps({"failure_count": len(failures), "output": str(args.output)}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
