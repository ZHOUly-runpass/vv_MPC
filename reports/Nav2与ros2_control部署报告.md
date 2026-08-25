# Nav2 与 ros2_control 部署报告

## 结论

开发机的 ROS 2 Humble Nav2、ros2_control、ros2_controllers 与 gazebo_ros2_control 已完成安装、构建和空场景导航闭环验收。`NavigateToPose` 的 1 m 目标返回 `SUCCEEDED`。

## 环境与安装验证

- 开发机：Ubuntu 22.04 / ROS 2 Humble。
- `nav2_bringup`、`controller_manager`、`diff_drive_controller`、`gazebo_ros2_control` 均解析到 `/opt/ros/humble`。
- 项目提交基线：`0e15f65`；本报告记录其后的 Nav2/ros2_control 适配修复。
- `r680_sim_description`、`r680_sim_worlds`、`r680_sim_bringup` 三个 ROS 包重新构建成功。

## 联合运行结果

- `joint_state_broadcaster`：`active`。
- `diff_drive_base_controller`：`active`。
- `map_server`、`controller_server`、`planner_server`、`bt_navigator`：全部为生命周期状态 `active [3]`。
- `/cmd_vel_nav`、`/cmd_vel`、`/odom`、`/scan`、`/map` 与控制器话题在线。
- 1 m `NavigateToPose` 目标：`Goal finished with status: SUCCEEDED`。
- 8 秒采样窗口：DWB 原始线速度峰值 `0.4642857 m/s`，平滑后线速度峰值 `0.4642857 m/s`，角速度峰值 `0.3 rad/s`，接收里程计 385 帧，`x` 方向前进 `0.6435219 m`。
- Gazebo 空场景的 `/scan` 全部为无穷远量程，符合无障碍环境预期。

## 实启发现并修复的问题

1. 原 Xacro 将 `if/unless` 写成普通 URDF 元素属性，Gazebo 同时加载旧差速插件和 gazebo_ros2_control。现改为 `xacro:if`/`xacro:unless` 包裹，确保两套驱动互斥。
2. 初始 DWB 参数缺少 `max_speed_xy` 等速度空间参数，导致 `/cmd_vel_nav` 只有角速度、线速度恒为零。现补齐 Humble DWB 的速度、采样、粒度和 critic 参数。
3. Humble 的导航启动会启用 velocity_smoother，现补齐与 R680 仿真限速一致的速度、加速度和超时参数。
4. diff_drive_controller 原生里程计位于 `/diff_drive_base_controller/odom`。现由底盘桥接节点转发至标准 `/odom`，供 Nav2 与后续安全规划模块统一使用。

## 尚未覆盖

- 本次只证明空场景 DWB + ros2_control 的安装与基础闭环正确，不等于八场景基线评测完成。
- MPPI 尚未安装和审计。
- Vanilla D-CBF、Proposed 方法尚未接入同一基准闭环。
- R680 尺寸、轮径、轮距和动力学参数仍为仿真假设，实车解锁前必须测量标定。
