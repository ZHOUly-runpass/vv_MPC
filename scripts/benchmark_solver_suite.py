#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys

import casadi
import numpy as np

from r680_safety_planner.dcbf import CasadiDcbfSolver
from r680_safety_planner.interfaces import MpcRequest, PredictedObstacle
from r680_safety_planner.planning import CandidateGenerator
from r680_safety_planner.vehicle import DifferentialModel, VehicleLimits


DEFAULT_COUNTS = (0, 8, 16, 32)


def git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def optional_finite(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def build_obstacles(count: int, horizon_points: int, placement: str) -> tuple[PredictedObstacle, ...]:
    if count < 0:
        raise ValueError("obstacle count must be non-negative")
    obstacles = []
    for index in range(count):
        states = np.zeros((horizon_points, 6), dtype=np.float64)
        if placement == "far":
            states[:, 0] = 4.0 + 0.25 * index
            states[:, 1] = 1.5 if index % 2 else -1.5
        elif placement == "near":
            states[:, 0] = 1.2 + 0.02 * index
            states[:, 1] = 0.8 if index % 2 else -0.8
        else:
            raise ValueError(f"unsupported placement: {placement}")
        states[:, 5] = 1.0
        obstacles.append(PredictedObstacle(
            states=states,
            lengths=np.full(horizon_points, 0.5, dtype=np.float64),
            widths=np.full(horizon_points, 0.5, dtype=np.float64),
            covariance=np.tile(np.eye(2, dtype=np.float64) * 0.01, (horizon_points, 1, 1)),
            valid_mask=np.ones(horizon_points, dtype=np.bool_),
        ))
    return tuple(obstacles)


def run_case(
    model: DifferentialModel,
    initial: np.ndarray,
    reference,
    obstacle_count: int,
    repeats: int,
    placement: str,
    deadline_ms: float,
) -> dict[str, object]:
    obstacles = build_obstacles(obstacle_count, reference.states.shape[0], placement)
    request = MpcRequest(initial, reference, obstacles)
    solver = CasadiDcbfSolver(model, 0.3, hard_deadline_ms=120_000.0)
    active_obstacles = len(solver.select_reachable_obstacles(request))
    results = [solver.solve(request) for _ in range(repeats)]
    times = [result.solve_time_ms for result in results]
    warm = times[1:] if len(times) > 1 else times
    finite_outputs = all(
        np.all(np.isfinite(result.states)) and np.all(np.isfinite(result.controls))
        for result in results
    )
    statuses = Counter(result.status for result in results)
    return {
        "obstacles": obstacle_count,
        "active_obstacles": active_obstacles,
        "all_obstacles_active": active_obstacles == obstacle_count,
        "repeats": repeats,
        "cold_ms": times[0],
        "warm_min_ms": min(warm),
        "warm_median_ms": statistics.median(warm),
        "warm_p95_ms": float(np.percentile(warm, 95)),
        "warm_max_ms": max(warm),
        "deadline_ms": deadline_ms,
        "deadline_pass": max(warm) <= deadline_ms,
        "feasible_runs": sum(result.feasible for result in results),
        "finite_outputs": finite_outputs,
        "status_counts": dict(sorted(statuses.items())),
        "h_min": optional_finite(min(result.h_min for result in results)),
        "slack_max": optional_finite(max(result.slack_max for result in results)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--obstacle-counts", type=int, nargs="+", default=DEFAULT_COUNTS)
    parser.add_argument("--repeats", type=int, default=11, help="one cold run plus warm runs")
    parser.add_argument("--deadline-ms", type=float, default=80.0)
    parser.add_argument("--placement", choices=("far", "near"), default="near")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least 2 to separate cold and warm timings")
    if tuple(args.obstacle_counts) != tuple(sorted(set(args.obstacle_counts))):
        parser.error("--obstacle-counts must be unique and ascending")

    root = Path(__file__).resolve().parents[1]
    limits = VehicleLimits(0.5, 0.1, 0.0, 1.0, 0.5, 0.8, 0.0, 1.0)
    model = DifferentialModel(limits)
    initial = np.zeros(5, dtype=np.float64)
    reference = CandidateGenerator(model, horizon_s=2.0, dt_s=0.1).generate(initial)[0]
    cases = [
        run_case(model, initial, reference, count, args.repeats, args.placement, args.deadline_ms)
        for count in args.obstacle_counts
    ]
    suite_valid = all(
        case["all_obstacles_active"]
        and case["finite_outputs"]
        and case["feasible_runs"] == case["repeats"]
        and set(case["status_counts"]) == {"Solve_Succeeded"}
        for case in cases
    )
    report = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(root),
        "environment": {
            "python": platform.python_version(),
            "executable": sys.executable,
            "casadi": casadi.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "benchmark": {
            "obstacle_counts": args.obstacle_counts,
            "placement": args.placement,
            "repeats_per_case": args.repeats,
            "horizon_s": 2.0,
            "dt_s": 0.1,
            "intervals": int(reference.controls.shape[0]),
            "deadline_ms": args.deadline_ms,
        },
        "cases": cases,
        "suite_valid": suite_valid,
        "all_deadlines_pass": all(case["deadline_pass"] for case in cases),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    print(rendered)
    if args.output is not None:
        output = args.output if args.output.is_absolute() else root / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if suite_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
