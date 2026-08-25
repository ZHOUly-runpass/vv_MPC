#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics

import numpy as np

from r680_safety_planner.dcbf import CasadiDcbfSolver
from r680_safety_planner.interfaces import MpcRequest, PredictedObstacle
from r680_safety_planner.planning import CandidateGenerator
from r680_safety_planner.vehicle import DifferentialModel, VehicleLimits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--deadline-ms", type=float, default=80.0)
    parser.add_argument("--obstacles", type=int, default=0)
    parser.add_argument("--placement", choices=("far", "near"), default="far")
    args = parser.parse_args()
    limits = VehicleLimits(0.5, 0.1, 0.0, 1.0, 0.5, 0.8, 0.0, 1.0)
    model = DifferentialModel(limits)
    initial = np.zeros(5, dtype=np.float64)
    reference = CandidateGenerator(model, horizon_s=2.0, dt_s=0.1).generate(initial)[0]
    horizon_points = reference.states.shape[0]
    obstacles = []
    for index in range(args.obstacles):
        states = np.zeros((horizon_points, 6), dtype=np.float64)
        if args.placement == "far":
            states[:, 0] = 4.0 + 0.25 * index
            states[:, 1] = 1.5 if index % 2 else -1.5
        else:
            states[:, 0] = 1.2 + 0.02 * index
            states[:, 1] = 0.8 if index % 2 else -0.8
        states[:, 5] = 1.0
        obstacles.append(PredictedObstacle(
            states=states,
            lengths=np.full(horizon_points, 0.5),
            widths=np.full(horizon_points, 0.5),
            covariance=np.tile(np.eye(2) * 0.01, (horizon_points, 1, 1)),
            valid_mask=np.ones(horizon_points, dtype=np.bool_),
        ))
    times = []
    statuses = []
    active_obstacles = None
    solver = CasadiDcbfSolver(model, 0.3, hard_deadline_ms=10000.0)
    for _ in range(args.repeats):
        request = MpcRequest(initial, reference, tuple(obstacles))
        active_obstacles = len(solver.select_reachable_obstacles(request))
        result = solver.solve(request)
        times.append(result.solve_time_ms)
        statuses.append(result.status)
    warm = times[1:] if len(times) > 1 else times
    report = {
        "repeats": len(times), "obstacles": args.obstacles,
        "active_obstacles": active_obstacles, "placement": args.placement,
        "cold_ms": times[0],
        "warm_min_ms": min(warm), "warm_median_ms": statistics.median(warm),
        "warm_p95_ms": float(np.percentile(warm, 95)), "deadline_ms": args.deadline_ms,
        "deadline_pass": max(warm) <= args.deadline_ms,
        "statuses": sorted(set(statuses)),
    }
    print(json.dumps(report, indent=2))
    return 0 if all(status == "Solve_Succeeded" for status in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
