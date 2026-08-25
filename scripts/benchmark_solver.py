#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics

import numpy as np

from r680_safety_planner.dcbf import CasadiDcbfSolver
from r680_safety_planner.interfaces import MpcRequest
from r680_safety_planner.planning import CandidateGenerator
from r680_safety_planner.vehicle import DifferentialModel, VehicleLimits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--deadline-ms", type=float, default=80.0)
    args = parser.parse_args()
    limits = VehicleLimits(0.5, 0.1, 0.0, 1.0, 0.5, 0.8, 0.0, 1.0)
    model = DifferentialModel(limits)
    initial = np.zeros(5, dtype=np.float64)
    reference = CandidateGenerator(model, horizon_s=2.0, dt_s=0.1).generate(initial)[0]
    times = []
    statuses = []
    for _ in range(args.repeats):
        solver = CasadiDcbfSolver(model, 0.3, hard_deadline_ms=10000.0)
        result = solver.solve(MpcRequest(initial, reference, ()))
        times.append(result.solve_time_ms)
        statuses.append(result.status)
    warm = times[1:] if len(times) > 1 else times
    report = {
        "repeats": len(times), "cold_ms": times[0],
        "warm_min_ms": min(warm), "warm_median_ms": statistics.median(warm),
        "warm_p95_ms": float(np.percentile(warm, 95)), "deadline_ms": args.deadline_ms,
        "deadline_pass": max(warm) <= args.deadline_ms,
        "statuses": sorted(set(statuses)),
    }
    print(json.dumps(report, indent=2))
    return 0 if all(status == "Solve_Succeeded" for status in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
