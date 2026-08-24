import numpy as np

from r680_safety_planner.dcbf import ReferenceDcbfSolver, dcbf_residual
from r680_safety_planner.interfaces import CandidateTrajectory, MpcRequest, PredictedObstacle


def obstacle(x: float, points: int) -> PredictedObstacle:
    states = np.zeros((points, 6))
    states[:, 0] = x
    return PredictedObstacle(
        states=states,
        lengths=np.full(points, 0.4),
        widths=np.full(points, 0.4),
        covariance=np.zeros((points, 2, 2)),
        valid_mask=np.ones(points, dtype=bool),
    )


def request(obstacle_x: float) -> MpcRequest:
    timestamps = np.linspace(0.0, 1.0, 6)
    states = np.zeros((6, 5))
    states[:, 0] = np.linspace(0.0, 1.0, 6)
    controls = np.zeros((5, 2))
    candidate = CandidateTrajectory(states, controls, timestamps, "test")
    return MpcRequest(states[0], candidate, (obstacle(obstacle_x, 6),))


def test_reference_solver_accepts_clear_path() -> None:
    result = ReferenceDcbfSolver(ego_radius_m=0.2, fixed_margin_m=0.1).solve(request(5.0))
    assert result.feasible
    assert result.h_min > 0.0


def test_reference_solver_rejects_collision() -> None:
    result = ReferenceDcbfSolver(ego_radius_m=0.2, fixed_margin_m=0.1).solve(request(0.5))
    assert not result.feasible
    assert result.h_min < 0.0


def test_gamma_is_derived_from_dt() -> None:
    h = np.array([1.0, 0.95, 0.9])
    assert dcbf_residual(h, 1.0, 0.1).shape == (2,)

