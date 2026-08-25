import numpy as np
from r680_safety_planner.dcbf import CasadiDcbfSolver
from r680_safety_planner.interfaces import MpcRequest, PredictedObstacle
from r680_safety_planner.planning import CandidateGenerator
from r680_safety_planner.vehicle import DifferentialModel, VehicleLimits


def obstacle(x: float, points: int, valid: bool = True) -> PredictedObstacle:
    states = np.zeros((points, 6))
    states[:, 0] = x
    return PredictedObstacle(states, np.full(points, 0.5), np.full(points, 0.5),
                             np.zeros((points, 2, 2)), np.full(points, valid))


def test_reachable_screen_is_conservative() -> None:
    limits = VehicleLimits(0.5, 0.1, 0.0, 1.0, 0.5, 0.8, 0.0, 1.0)
    model = DifferentialModel(limits)
    reference = CandidateGenerator(model, 2.0, 0.1).generate(np.zeros(5))[0]
    points = reference.states.shape[0]
    near, far, invalid = obstacle(1.0, points), obstacle(5.0, points), obstacle(5.0, points, False)
    request = MpcRequest(np.zeros(5), reference, (near, far, invalid))
    selected = CasadiDcbfSolver(model, 0.3).select_reachable_obstacles(request)
    assert any(item is near for item in selected)
    assert not any(item is far for item in selected)
    assert any(item is invalid for item in selected)
