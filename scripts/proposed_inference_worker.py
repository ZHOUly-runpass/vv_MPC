#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from r680_safety_planner.data import load_feature_cache, load_teacher_vehicle_config
from r680_safety_planner.dcbf import CasadiDcbfSolver
from r680_safety_planner.interfaces import CandidateTrajectory, MpcRequest, PredictedObstacle
from r680_safety_planner.learned_runtime import (
    LearnedPlannerRuntime, file_sha256, ranked_candidate_indices, validate_checkpoint_contract,
    validate_inference_inputs,
)
from r680_safety_planner.models import PlanningSafetyModel


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def obstacles_from_metadata(items: list[dict], timestamps: np.ndarray) -> tuple[PredictedObstacle, ...]:
    obstacles = []
    for item in items:
        if not item.get("collision_check", True):
            continue
        position = np.asarray(item["position"][:2], dtype=np.float64)
        velocity = np.asarray(item["velocity"][:2], dtype=np.float64)
        centers = position[None] + timestamps[:, None] * velocity[None]
        states = np.zeros((timestamps.size, 6), dtype=np.float64)
        states[:, :2] = centers; states[:, 3:5] = velocity; states[:, 5] = 1.0
        radius = float(item["radius_m"]); side = np.sqrt(2.0) * radius
        covariance = np.asarray([np.eye(2) * (0.0025 + 0.02 * value) for value in timestamps])
        obstacles.append(PredictedObstacle(
            states, np.full(timestamps.size, side), np.full(timestamps.size, side),
            covariance, np.ones(timestamps.size, dtype=np.bool_), source="simulation_ground_truth",
        ))
    return tuple(obstacles)


def main() -> int:
    parser = argparse.ArgumentParser(description="Isolated Proposed planner inference and D-CBF worker")
    parser.add_argument("--request-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-version", type=Path, required=True)
    parser.add_argument("--unilion-checkpoint", type=Path, required=True)
    parser.add_argument("--vehicle-config", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--deadline-ms", type=float, default=80.0)
    parser.add_argument("--feature-timeout-s", type=float, default=0.30)
    parser.add_argument("--poll-ms", type=float, default=5.0)
    args = parser.parse_args()
    args.request_dir.mkdir(parents=True, exist_ok=True)
    ready_path = args.request_dir / "worker_status.json"
    result_path = args.request_dir / "latest_result.json"
    request_path = args.request_dir / "latest_request.npz"
    try:
        checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=True)
        contract = validate_checkpoint_contract(
            checkpoint, args.checkpoint, args.manifest, args.dataset_version, args.unilion_checkpoint,
            args.expected_checkpoint_sha256,
        )
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required but unavailable")
        model = PlanningSafetyModel(**checkpoint["model_config"]).to(args.device)
        model.load_state_dict(checkpoint["model_state"], strict=True); model.eval()
        runtime = LearnedPlannerRuntime(model, args.device, args.deadline_ms)
        vehicle = load_teacher_vehicle_config(args.vehicle_config); timestamps = np.arange(21) * vehicle.dt_s
        solver = CasadiDcbfSolver(
            vehicle.model, vehicle.ego_radius_m,
            fixed_margin_m=float(vehicle.mpc.get("fixed_margin_m", 0.1)),
            sigma_multiplier=float(vehicle.mpc.get("sigma_multiplier", 3.0)),
            continuous_alpha=float(vehicle.mpc.get("continuous_alpha", 1.0)),
            maximum_slack=float(vehicle.mpc.get("maximum_slack", 1.0)),
            hard_deadline_ms=args.deadline_ms, max_iterations=int(vehicle.mpc.get("max_iterations", 100)),
        )
        atomic_json(ready_path, {"ready": True, "contract": asdict(contract), "pid": os.getpid()})
    except Exception as error:
        atomic_json(ready_path, {"ready": False, "error": f"{type(error).__name__}:{error}"})
        return 2

    running = True
    def stop(*_):
        nonlocal running; running = False
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    last_sequence = -1
    while running:
        if not request_path.is_file():
            time.sleep(args.poll_ms / 1000.0); continue
        try:
            with np.load(request_path, allow_pickle=False) as values:
                metadata = json.loads(str(values["metadata_json"]))
                sequence = int(metadata["sequence"])
                if sequence == last_sequence:
                    time.sleep(args.poll_ms / 1000.0); continue
                route = values["route"].astype(np.float32); ego = values["ego"].astype(np.float32)
                costmap = values["costmap"].astype(np.float32)
            last_sequence = sequence
            cache_path = Path(metadata["feature_cache_path"])
            if file_sha256(cache_path) != metadata["feature_cache_sha256"]:
                raise ValueError("feature cache SHA-256 mismatch")
            feature = load_feature_cache(cache_path)
            if feature.checkpoint_hash != contract.unilion_checkpoint_sha256:
                raise ValueError("feature UniLION checkpoint SHA-256 mismatch")
            validate_inference_inputs(feature.bev_feature, route, ego, costmap,
                                      float(metadata["feature_age_s"]), args.feature_timeout_s)
            prediction, inference_ms = runtime.infer(feature.bev_feature, route, ego, costmap)
            obstacles = obstacles_from_metadata(metadata.get("obstacles", []), timestamps)
            controls = prediction["controls"][0]
            selected = None; result = None
            for index in ranked_candidate_indices(prediction):
                candidate_controls = controls[index].astype(np.float64)
                candidate = CandidateTrajectory(
                    vehicle.model.rollout(ego.astype(np.float64), candidate_controls, vehicle.dt_s),
                    candidate_controls, timestamps.astype(np.float64), role=f"learned_{index}",
                )
                result = solver.solve(MpcRequest(ego.astype(np.float64), candidate, obstacles), initial_guess=candidate)
                if result.feasible and inference_ms + result.solve_time_ms <= args.deadline_ms:
                    selected = int(index); break
            if selected is None or result is None:
                raise RuntimeError("no learned candidate passed D-CBF and deadline checks")
            command = vehicle.model.command(ego.astype(np.float64), result.controls[0], float(metadata["stamp_s"]))
            values = np.asarray([command.longitudinal_velocity, command.lateral_velocity, command.yaw_rate])
            if not np.all(np.isfinite(values)):
                raise ValueError("final command contains NaN or Inf")
            atomic_json(result_path, {
                "sequence": sequence, "ok": True, "selected_index": selected,
                "command": values.tolist(), "inference_ms": inference_ms,
                "mpc_solve_time_ms": result.solve_time_ms, "h_min": result.h_min,
                "slack_max": result.slack_max, "mpc_status": result.status,
                "completed_wall_time_s": time.time(), "contract": asdict(contract),
            })
        except Exception as error:
            atomic_json(result_path, {"sequence": last_sequence, "ok": False,
                                      "error": f"{type(error).__name__}:{error}",
                                      "completed_wall_time_s": time.time()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
