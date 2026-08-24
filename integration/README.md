# ROS 2 开发机验证

开发机已确认是 Ubuntu 22.04 / ROS 2 Humble。所有运动测试前，保持急停可触达、车轮离地，且配置中的验证门不得批量改为 `true`。

```bash
cd <项目05目录>
bash scripts/dev_machine_build.sh "$PWD"
source /opt/ros/humble/setup.bash
source .venv/bin/activate
source integration/ros2_ws/install/setup.bash
python scripts/audit_ros2_runtime.py \
  --config configs/robot/r680_c16.yaml \
  --output reports/runtime_ros_graph.json
ros2 launch r680_safety_planner_ros planner.launch.py \
  config:="$PWD/configs/robot/r680_c16.yaml"
```

当前配置是 `perception_only`，节点应持续发布零速度。只有在逐项实测并记录证据后，才能修改对应验证门。启动前必须确认 `/cmd_vel` 没有其他发布者；生产部署应由唯一的安全监督节点拥有最终控制话题。
