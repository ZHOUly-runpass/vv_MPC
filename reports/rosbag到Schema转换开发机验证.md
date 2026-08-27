# rosbag 到 schema 1.0 转换开发机验证

日期：2026-08-27

## 验证输入

- 场景：`empty`
- 控制器：DWB闭环
- seed：47
- 难度：nominal
- 录制时长：15秒
- 抽样频率：1 Hz（流水线默认2 Hz）
- UniLION存储网格：32 × 32

## 验证结果

1. rosbag成功录制传感器、TF、odom、路线、costmap、控制器状态，以及候选轨迹、障碍预测、MPC request/result四个规划追踪话题。本次bag包含141帧点云、22帧costmap、28条控制器状态，以及四类规划追踪话题各62条。
2. 纯Python `rosbags 0.10.11` 成功读取 Humble sqlite3 bag，并注册解析 `nav2_msgs/Costmap`。
3. 提取13个同步原始训练帧，源bag SHA-256为 `ac19e498fbf9a381f15bb0b35b32835f0a3068da4e3371c70515c751142037f6`。
4. UniLION对13帧完成离线前向，输出13份 `[384,32,32]` 特征。
5. 使用的LiDAR backbone checkpoint SHA-256为 `9ba65cfead901ea08db3664108838d4c85bd6621592cc08cca0dfe61af4eb262`。
6. 13个样本全部组装为schema 1.0，并被离线CasADi teacher成功读取和标注。
7. 最终审计为13/13有效，统一形状为点字段6维、特征 `[384,32,32]`、路线 `[32,4]`、costmap `[3,60,60]`、候选 `[7,11,5]`；91条候选teacher结果均为success。
8. 另用Proposed入口录制8秒闭环bag，得到70条 `/cmd_vel`、11帧Nav2 costmap和34条候选轨迹；转换器成功提取7个同步原始样本，证明自定义控制器分支也具备完整costmap。

本次验证证明数据工程链路可执行，不代表空场景数据具备训练价值，也不代表真实C16已经通过验证。
