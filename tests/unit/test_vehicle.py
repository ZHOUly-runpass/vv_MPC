import numpy as np

from r680_safety_planner.planning import CandidateGenerator
from r680_safety_planner.vehicle import AckermannModel, DifferentialModel, MecanumModel, VehicleLimits


def limits() -> VehicleLimits:
    return VehicleLimits(1.0, 0.2, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0)


def test_differential_can_rotate_in_place() -> None:
    model = DifferentialModel(limits())
    state = np.zeros(5)
    controls = np.tile([0.0, 1.0], (10, 1))
    result = model.rollout(state, controls, 0.1)
    assert result[-1, 2] > 0.0
    assert np.allclose(result[:, :2], 0.0)


def test_mecanum_lateral_motion() -> None:
    model = MecanumModel(limits())
    result = model.rollout(np.zeros(6), np.tile([0.0, 0.5, 0.0], (10, 1)), 0.1)
    assert result[-1, 1] > 0.0


def test_ackermann_command_converts_steering_to_yaw_rate() -> None:
    model = AckermannModel(limits(), wheelbase_m=0.5)
    command = model.command(np.array([0.0, 0.0, 0.0, 0.5]), np.array([0.0, 0.2]), 1.0)
    assert command.angular_z > 0.0
    assert command.linear_y == 0.0


def test_candidate_generator_produces_seven_valid_candidates() -> None:
    candidates = CandidateGenerator(DifferentialModel(limits())).generate(np.zeros(5))
    assert len(candidates) == 7
    assert {candidate.role for candidate in candidates} == set(CandidateGenerator.ROLES)


def test_nominal_candidate_accelerates_toward_route() -> None:
    model = DifferentialModel(limits())
    generator = CandidateGenerator(model, horizon_s=1.0, dt_s=0.1)
    route = np.array([[0.0, 0.0], [2.0, 0.5]])
    nominal = generator.generate(np.zeros(5), route_xy=route)[0]
    assert nominal.states[-1, 0] > 0.0
    assert nominal.states[-1, 2] > 0.0
