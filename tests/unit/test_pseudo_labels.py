import numpy as np

from r680_safety_planner.data.pseudo_labels import classify_solver_result
from r680_safety_planner.interfaces import MpcResult


def result(feasible=True, time=10.0, status="Solve_Succeeded"):
    return MpcResult(np.zeros((2, 4)), np.zeros((1, 2)), feasible, 1.0, 0.0, time, status)


def test_solver_outcomes_are_not_conflated() -> None:
    assert classify_solver_result(result(), 80.0) == "feasible"
    assert classify_solver_result(result(False), 80.0) == "infeasible"
    assert classify_solver_result(result(False, 90.0), 80.0) == "timeout"
    assert classify_solver_result(result(False, status="numeric_failure"), 80.0) == "numeric_failure"
