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

Available scenarios are listed in `config/scenarios.yaml`. `difficulty:=easy|nominal|hard`
now changes static obstacle count, passage width, start/goal placement or trap
safety spacing; it is not only a metadata label. The launch provides
`/clock`, `/tf`, `/tf_static`, `/odom`, `/imu/data_raw`, `/scan`, `/points_raw`,
adapted `/points`, `/plan`, `/cmd_vel`, ground-truth obstacle poses and benchmark
status. 普通场景启动时 `/plan` 由确定性路线发布器提供；Nav2 联合启动时由
Nav2 规划器提供。

Every launch accepts a deterministic seed. Record a repeatable run into the ignored
project-local `.tools/rosbags` directory with:

```bash
bash simulation/scripts/record_scenario.sh "$PWD" static_sparse 60 42 hard dwb
```

Before rosbag starts, the recorder fails closed unless its preflight confirms
required topics, minimum observed frequencies, `odom -> base_footprint` and
`base_footprint -> c16_link` TF, scenario difficulty readiness, and at least
5 GiB free disk. The JSON evidence is written under `.tools/`.

`/simulation/reset_benchmark` clears metrics and requests Gazebo's
`/reset_simulation` service. DWB 已完成安装和空场景闭环验收；MPPI 尚未安装和审计。

Run the Stage-3 100-frame capture and isolated UniLION validation with:

```bash
bash simulation/scripts/run_stage3_validation.sh "$PWD"
```

The verified DWB/ros2_control stack starts with:

```bash
ros2 launch r680_sim_bringup nav2_sim.launch.py scenario:=empty controller:=dwb
```
