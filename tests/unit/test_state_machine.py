from r680_safety_planner.execution import StartupEvidence, StartupState, StartupStateMachine


def test_startup_requires_every_stage_and_manual_arm() -> None:
    machine = StartupStateMachine()
    assert machine.update(StartupEvidence()) is StartupState.SELF_TEST_GUARD
    assert machine.update(StartupEvidence(self_test_elapsed=True)) is StartupState.SENSOR_CHECK
    assert machine.update(StartupEvidence(sensors_fresh=True)) is StartupState.FRAME_CHECK
    assert machine.update(StartupEvidence(frames_verified=True)) is StartupState.PERCEPTION_ONLY
    assert machine.update(StartupEvidence(all_validation_gates=True)) is StartupState.PERCEPTION_ONLY
    assert machine.update(StartupEvidence(all_validation_gates=True, manual_arm=True)) is StartupState.ARMED


def test_fault_latches_until_reset() -> None:
    machine = StartupStateMachine()
    assert machine.update(StartupEvidence(fault=True)) is StartupState.FAULT_STOP
    assert machine.update(StartupEvidence(all_validation_gates=True, manual_arm=True)) is StartupState.FAULT_STOP
    machine.reset_fault()
    assert machine.state is StartupState.BOOT
