# R680 Humble + Gazebo Classic simulation

This workspace contains an explicitly approximate differential-drive R680
model. Dimensions and dynamics are simulation assumptions, not measured robot
parameters and must not be copied into the real-vehicle safety profile.

```bash
cd /home/zhou/E2Eproject_MPC/github_pull/05
bash simulation/scripts/build_simulation.sh "$PWD"
source /opt/ros/humble/setup.bash
source simulation/ros2_ws/install/setup.bash
ros2 launch r680_sim_bringup simulation.launch.py scenario:=empty headless:=true
```

Available scenarios are listed in `config/scenarios.yaml`. The launch provides
`/clock`, `/tf`, `/tf_static`, `/odom`, `/imu/data_raw`, `/scan`, `/points_raw`,
adapted `/points`, `/plan`, `/cmd_vel`, ground-truth obstacle poses and benchmark
status. Nav2 is not installed on the current machine, so `/plan` is initially
provided by the deterministic route publisher.
