#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from r680_safety_planner.dcbf import CasadiDcbfSolver
from r680_safety_planner.interfaces import MpcRequest
from r680_safety_planner.planning import CandidateGenerator
from r680_safety_planner.vehicle import DifferentialModel, VehicleLimits


def main() -> int:
    limits = VehicleLimits(0.5, 0.1, 0.0, 1.0, 0.5, 0.8, 0.0, 1.0)
    model = DifferentialModel(limits)
    initial = np.zeros(5, dtype=np.float64)
    reference = CandidateGenerator(model, horizon_s=1.0, dt_s=0.2).generate(initial)[0]
    solver = CasadiDcbfSolver(model, ego_radius_m=0.3, hard_deadline_ms=10000.0)
    result = solver.solve(MpcRequest(initial, reference, ()))
    summary = {
        "available": solver.available(), "feasible": result.feasible,
        "status": result.status, "solve_time_ms": result.solve_time_ms,
        "finite": bool(np.all(np.isfinite(result.states))),
    }
    print(summary)
    if not result.feasible or not summary["finite"]:
        raise RuntimeError("CasADi smoke test failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
