# R680 C16 Safety Planner

Implementation of the workflow in [算法完整流程.md](算法完整流程.md). The local
Windows environment runs the dependency-light core and tests. ROS 2 integration
is isolated under `integration/ros2_ws` and is validated on the Linux
development machine.

Safety defaults:

- configuration starts in `perception_only`;
- non-zero commands are forbidden until every commissioning gate passes;
- stale LiDAR, odometry, IMU, TF, planner heartbeat, or commands force zero;
- geometric obstacles remain active independently of learned perception;
- unresolved vehicle type or footprint prevents motion.

Local setup:

```powershell
python -m pip install -e ".[test]"
python -m pytest
python scripts/validate_config.py --config configs/robot/r680_c16.yaml
python scripts/run_synthetic_smoke.py --config configs/robot/r680_c16.yaml
```

ROS 2 development-machine setup and the fail-closed adapter are described in
`integration/README.md`.
