import numpy as np

from r680_safety_planner.execution import CommandAdapter, SafetyInputs, SafetySupervisor
from r680_safety_planner.interfaces import ExecutionCommand
from r680_safety_planner.vehicle import VehicleLimits


LIMITS = VehicleLimits(1.0, 0.2, 0.0, 1.0, 0.5, 1.0, 0.0, 1.0)


def supervisor(unlocked: bool) -> SafetySupervisor:
    return SafetySupervisor(CommandAdapter("differential", LIMITS, unlocked), 0.2, 0.3, 0.15, 0.1, 0.1, 0.1)


def inputs(now: float = 1.0) -> SafetyInputs:
    return SafetyInputs(now, ExecutionCommand(0.5, 0.0, 0.2, now, "test"), now, now, now, now, now)


def test_locked_supervisor_forces_zero() -> None:
    decision = supervisor(False).evaluate(inputs())
    assert not decision.allowed
    assert np.allclose(decision.command.as_array(), 0.0)
    assert "motion_locked" in {event.code for event in decision.events}


def test_unlocked_fresh_command_is_allowed() -> None:
    decision = supervisor(True).evaluate(inputs())
    assert decision.allowed
    assert decision.command.linear_x == 0.5


def test_stale_lidar_forces_zero() -> None:
    item = inputs()
    stale = SafetyInputs(**{**item.__dict__, "point_cloud_s": 0.0})
    decision = supervisor(True).evaluate(stale)
    assert not decision.allowed
    assert "stale_point_cloud" in {event.code for event in decision.events}

