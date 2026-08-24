from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class StartupState(str, Enum):
    BOOT = "boot"
    SELF_TEST_GUARD = "self_test_guard"
    SENSOR_CHECK = "sensor_check"
    FRAME_CHECK = "frame_check"
    PERCEPTION_ONLY = "perception_only"
    ARMED = "armed"
    FAULT_STOP = "fault_stop"


@dataclass(frozen=True)
class StartupEvidence:
    self_test_elapsed: bool = False
    sensors_fresh: bool = False
    frames_verified: bool = False
    all_validation_gates: bool = False
    manual_arm: bool = False
    fault: bool = False


class StartupStateMachine:
    """Monotonic startup gate; faults latch until a deliberate reset."""

    def __init__(self) -> None:
        self.state = StartupState.BOOT

    @property
    def motion_allowed(self) -> bool:
        return self.state is StartupState.ARMED

    def update(self, evidence: StartupEvidence) -> StartupState:
        if evidence.fault:
            self.state = StartupState.FAULT_STOP
            return self.state
        if self.state is StartupState.FAULT_STOP:
            return self.state
        if self.state is StartupState.BOOT:
            self.state = StartupState.SELF_TEST_GUARD
        if self.state is StartupState.SELF_TEST_GUARD and evidence.self_test_elapsed:
            self.state = StartupState.SENSOR_CHECK
        if self.state is StartupState.SENSOR_CHECK and evidence.sensors_fresh:
            self.state = StartupState.FRAME_CHECK
        if self.state is StartupState.FRAME_CHECK and evidence.frames_verified:
            self.state = StartupState.PERCEPTION_ONLY
        if (self.state is StartupState.PERCEPTION_ONLY and evidence.all_validation_gates
                and evidence.manual_arm):
            self.state = StartupState.ARMED
        return self.state

    def reset_fault(self) -> None:
        if self.state is StartupState.FAULT_STOP:
            self.state = StartupState.BOOT
