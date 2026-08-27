# R680/C16 VAD-MPC 项目执行计划

## 阶段 4 本轮实现状态（2026-08-27）

1. `[已完成：代码]` 版本化训练样本格式、完整字段、严格校验和 SHA-256 载荷审计。
2. `[已完成：代码]` 8 场景 × 3 种子 × 3 难度批量采集矩阵；完整 72 次真实采集尚未执行。
3. `[已完成：代码]` CasADi MPC teacher 离线生成，严格区分成功、不可行、数值失败和超时。
4. `[已完成：代码]` Dataset、DataLoader、联合损失、训练、checkpoint 和评测程序。
5. `[已完成：统一入口]` DWB、MPPI、Vanilla D-CBF、Proposed 四组闭环启动接口；真实数据训练 checkpoint 和完整公平批量评测仍待执行。
6. `[已完成：代码与开发机实测]` rosbag → 同步原始帧 → UniLION离线特征 → schema 1.0 NPZ → CasADi teacher流水线。
7. `[已完成：代码]` 录制入口支持四控制器，并记录控制器状态、候选轨迹、障碍预测、MPC request/result。
8. `[已完成：代码与开发机实测]` 静态难度会改变障碍数量、通道宽度、起终点或安全间距；`static_sparse` easy/hard 实测分别得到 1/3 个有效障碍。
9. `[已完成：代码与开发机实测]` 每次录制前强制检查 topic、频率、TF、场景难度状态和剩余磁盘；easy/hard 两次录制均通过。
10. `[已完成：代码与开发机实测]` 候选与 MPC 统一为 `2.0 s / 0.1 s / 21状态点`；旧 `0.2 s` 数据按“控制零阶保持、状态重新 rollout、障碍线性插值”转换。
11. `[已完成：安全门]` teacher 改为读取版本化车辆配置；仿真读取 `r680_sim.yaml`，未确认的 R680 实车配置会立即拒绝生成标签。

更新日期：2026-08-25  
本地目录：`D:\E2Eproject_MPC\05`  
开发机目录：`/home/zhou/E2Eproject_MPC/github_pull/05`

# 第一部分：已经完成的内容

## 1. 项目工程与算法骨架

- 已建立独立的 `05` 项目，代码、配置、脚本、测试、文档和报告均位于该目录内。
- 已实现配置加载、车辆约束、候选轨迹生成、障碍物筛选、D-CBF 约束、CasADi MPC 求解、安全监督、启动状态机、命令适配和 watchdog 等基础模块。
- 已实现 ROS 2 Humble 包、launch 文件、ROS 图审计脚本、零速实启和故障注入测试入口。
- 已建立 feature cache、伪标签、感知几何基线以及训练/验证所需的数据接口骨架。
- 本地和开发机项目回归测试均通过，共 `26 passed`。

## 2. 规划与安全验证

- 开发机上 CasADi 求解状态为 `Solve_Succeeded`，输出状态保持有限。
- 参数化问题缓存后，20 步 MPC 热态耗时约为：1 个近场障碍 10 ms、8 个近场障碍 35 ms，满足当前 80 ms 控制周期目标。
- 32 个同时位于可达近场的障碍约为 122 ms；系统会按超时策略发布零速，尚未满足正常运行要求。
- 已实现严格可达域筛选，能够安全排除规划周期内不可能接触的远处障碍。
- 已验证 11 类故障均会触发停车，包括数据超时、TF 超时、求解超时、急停、硬件故障、紧急障碍和非有限命令。
- ROS 零速实启时，`/cmd_vel` 所有分量为零，系统保持 `commissioning_lock`，未提前解锁实车控制。

## 3. UniLION 代码、权重与开发机环境

- UniLION 已作为 Git submodule 纳入项目，固定官方提交为 `b75c28415ed53d5474bcb9be776d032c66ec432b`。
- 官方 LCT checkpoint 已下载到本地和开发机，大小 `400672958` 字节，SHA-256 为 `14377fe6656ed487f40ad6af3161055bcc68956394599845b1c6f234a1b41256`。
- 已从 LCT 权重中确定性抽取 voxel encoder、LiDAR backbone、BEV backbone 和 neck，共911项，SHA-256 为 `9ba65cfead901ea08db3664108838d4c85bd6621592cc08cca0dfe61af4eb262`。
- 完整权重和 LiDAR 子集均不提交 Git；本地与开发机哈希一致。
- 已在开发机项目内部建立隔离环境：Python 3.9、PyTorch 2.5.0+cu124、nvcc 12.4.131、MMCV 1.7.0、MMDetection3D 1.0.0rc6、Mamba 1.1.1、`causal-conv1d` 1.5.0.post8。
- RTX 4090 CUDA 可用，`mmcv._ext`、`mmdet3d`、`mamba_ssm` 和 `projects.mmdet3d_plugin` 导入测试全部通过。
- 已固化环境文件、自动部署脚本、checkpoint 抽取脚本、环境审计脚本和开发机冒烟测试脚本。

## 4. GitHub 与当前完成边界

- 项目已提交至 `https://github.com/ZHOUly-runpass/vv_MPC.git`；阶段成果以 Git 历史中的最新 `main` 提交为准，并在每阶段收尾时同步开发机。
- 当前“UniLION 已部署”仅表示代码、依赖、CUDA 扩展和插件导入已经就绪，不表示已经完成 C16 推理验收。
- 尚未发现公开的 C16 专用或经 C16 验证的纯 LiDAR checkpoint。当前抽取权重只能作为初始化候选，`frozen_backbone_verified_on_c16` 必须保持关闭。

# 第二部分：开发机上部署算法的框架

## 1. 环境与目录隔离

开发机采用“双环境、单项目、消息边界隔离”的结构：

```text
/home/zhou/E2Eproject_MPC/github_pull/05
├── src/、ros2_ws/、config/       ROS 2、规划、MPC、安全监督
├── third_party/UniLION/          固定版本的上游感知代码
├── artifacts/checkpoints/        本地权重和哈希元数据，不进入 Git
├── .tools/envs/unilion/          Python 3.9 / Torch 2.5 / CUDA 12.4
├── environments/                 可复现环境定义
├── scripts/                      部署、审计、测试和实启脚本
├── simulation/                   后续 Gazebo/Nav2 仿真工作区
└── reports/                      测试、性能和验收证据
```

- ROS 2 Humble、底盘接口、MPC 和安全监督继续使用 Ubuntu 22.04 的系统 ROS 环境。
- UniLION 使用项目内独立 Python 3.9/CUDA 12.4 环境，不向 ROS 系统 Python 安装修改版 MMCV。
- 两个进程通过稳定的 ROS 消息或落盘 feature cache 交互，避免 Python、Torch 和 CUDA 依赖互相污染。

## 2. 运行时算法链

```text
C16 / 仿真16线点云
        │
        ├── 几何安全通路：裁剪 → 聚类/跟踪 → 障碍物状态 ───────────┐
        │                                                        │
        └── 学习感知通路：点字段适配 → Frozen UniLION backbone    │
                                      → BEV/场景特征              │
                                                               ▼
全局路径 / 里程计 / IMU / TF ─→ 场景上下文 ─→ 多候选轨迹生成 ─→ 候选筛选
                                                               │
                                                               ▼
                                  可达域障碍筛选 → D-CBF-MPC 求解
                                                               │
                                                               ▼
                          安全监督 / watchdog / 限幅 / 状态机
                                                               │
                                                               ▼
                                                 /cmd_vel → R680底盘
```

其中几何通路是独立的确定性安全来源，学习 backbone 不能成为唯一停车依据。UniLION 异常、特征超时、MPC 超时或任一关键传感器失效时，安全监督直接发布零速。

## 3. ROS 2 节点和主要接口

计划中的运行节点划分如下：

- `lidar_preprocessor`：C16 PointCloud2 字段检查、坐标变换、时间同步、范围裁剪和降采样。
- `geometric_obstacle_node`：几何聚类、跟踪、速度估计和统一障碍物输出。
- `unilion_backbone_node`：在隔离环境中加载冻结 backbone，输出 BEV/场景特征和健康状态。
- `candidate_planner_node`：结合路径、车辆状态和场景特征生成多条候选轨迹。
- `dcbf_mpc_node`：可达域筛选、D-CBF 约束构造、MPC 优化与候选选择。
- `safety_supervisor_node`：检查心跳、时间戳、TF、有限值、急停、硬件故障和求解时限。
- `command_adapter_node`：速度/加速度限幅、底盘方向适配和最终 `/cmd_vel` 发布。
- `benchmark_manager`：用于仿真重置、随机种子、故障注入和指标记录。

主要输入输出暂定为 `/points`、`/scan`、`/odom`、`/imu/data_raw`、`/tf`、`/plan`、障碍物接口、冻结特征接口和 `/cmd_vel`。真实话题名、消息类型和 QoS 必须在 C16 与底盘驱动上线后确认。

## 4. 配置、启动与安全门

- 车辆尺寸、速度、加速度、MPC、D-CBF、传感器时限和话题映射统一由 YAML 配置管理，不在节点中硬编码。
- 启动顺序为：配置审计 → 传感器/TF 审计 → 零速 commissioning → 感知健康检查 → 规划 dry-run → 低速解锁。
- checkpoint 必须校验 SHA-256；字段、坐标系或网络结构不匹配时拒绝加载。
- 23 项实车 validation gates 在证据不完整时保持关闭，任何学习模块不得绕过安全监督直接控制底盘。

# 第三部分：后续计划

## 阶段 1：完成 UniLION 可执行前向（工程实现完成）

目标：从“插件可导入”推进到“LiDAR 子网可重复推理”。

1. `[已完成]` 使用官方配置构建单帧LiDAR路径，记录模型构建、显存和时延；开发机缺少nuScenes原始样例，因此真实样例项待数据补跑，当前输入明确标记为合成C16格式。
2. `[已完成]` 编写 `FrozenLidarBackbone`，严格加载抽取的 voxel encoder、LiDAR backbone、BEV backbone和neck权重，无 missing/unexpected keys。
3. `[已完成]` 明确模型输入为 `x,y,z,intensity,ring`，相对时间仅用于上游去畸变；内部11维由模型构造，不填伪值。
4. `[已完成]` 增加四级BEV hook、特征维度契约、有限值检查、重复性检查和单元测试。

完成结果：12000点输出 `[1,384,180,180]`，权重报告可审计，有限值和方差通过，模型/总耗时中位数约56.64/70.75 ms，峰值显存约491.5 MB，重复差 `3.72e-4 < 1e-3`。真实nuScenes样例与真实C16 100帧仍属于后续数据验收，不作为本次合成结果的替代。

## 阶段 2：搭建 Humble + Gazebo Classic 仿真

目标：先建立可重复的确定性闭环，不升级开发机宿主系统。

1. `[已完成]` 开发机已安装 Navigation2/Nav2 Bringup、ros2_control、ros2_controllers 和 gazebo_ros2_control；四个核心包均解析到 `/opt/ros/humble`。ros2_control URDF、控制器 YAML、标准 `/cmd_vel` 与 `/odom` 桥接、Nav2 DWB 参数和一键 launch 已完成实启验证。
2. `[已完成]` 建立 R680 近似差速 URDF/Xacro、惯导、720 线束 2D 雷达和 16 线 3D LiDAR；近似尺寸仅供仿真，不能替代实车标定。
3. `[已完成]` 开发机实测 `/cmd_vel → /odom → /tf`、`/scan`、`/points_raw → /points`、`/imu/data_raw` 和 `/plan` 链路在线；点云严格输出 `x/y/z/intensity/ring`。
4. `[已完成]` 实现并实启 Empty、Static Sparse、Static Dense、Narrow Passage、Crossing、Head-on、Multi Dynamic、Local Minimum Trap 八类场景。
5. `[已完成]` 实现 GT obstacle bridge、`/simulation/reset_benchmark`、固定随机种子 42、项目内 rosbag 录制脚本和 benchmark manager。
6. `[部分完成]` DWB 已完成空场景 `NavigateToPose` 闭环验证；MPPI、Vanilla D-CBF 和 Proposed 的公平批量评测仍需在下一步把相应规划节点接入仿真 `/cmd_vel` 后完成。

阶段 2 当前验收证据：开发机三个 ROS 2 包编译通过，八类场景运行审计全部 `passed=true`；Crossing 的 GT 障碍物 3 秒位移为 `0.576 m`，Head-on 和 Multi Dynamic 的最大位移为 `2.288 m`。空场景观测频率约为 odom 50 Hz、IMU 100 Hz、scan 15 Hz、16 线点云 10 Hz。Nav2 联合实测中两个控制器和四个核心生命周期节点均为 `active`，1 m 导航目标返回 `SUCCEEDED`，采样窗口里程计前进 `0.6435 m`。证据位于 `reports/simulation_*_runtime.json` 和 `reports/Nav2与ros2_control部署报告.md`。

完成标准：八类场景能够一键运行、重复、记录并汇总成功率、碰撞率、最小间距、到达时间、轨迹平滑度和计算延迟。

## 阶段 3：完成16线点云和特征桥接

目标：让仿真点云与未来 C16 数据使用同一适配接口。

1. `[已完成（仿真）]` 实现 `x/y/z/intensity/ring/time` 字段映射、仿真已知外参变换和时间同步；真实 C16 外参与原生时间字段仍必须实车标定。
2. `[已完成]` 运行 100 帧真实 Gazebo 16 线点云：720000 点、10 Hz、字段完整、有限值通过；空体素率、特征方差、显存和延迟均已记录。
3. `[已完成]` UniLION 在独立 Python 3.9/CUDA 进程运行，通过 schema 1.0、缓存 SHA-256、FrozenSceneFeatures 引用和健康状态与 ROS 进程隔离桥接。
4. `[已完成]` `/scan` 与几何 `/points` 通路保持独立；UniLION 退出、超时、缓存损坏或非有限输出会触发 feature stop 和零速。

阶段 3 仿真验收结果：100 帧全部输出 `[1,384,180,180]`，特征方差中位数 `0.00104578`，总耗时中位数 `61.17 ms`、P95 `61.72 ms`，最大显存 `474784256` 字节。报告位于 `reports/阶段3执行报告.md`、`reports/simulation_c16_100_frames.json` 和 `reports/unilion_stage3_sim100.json`。真实 C16 仍未验收，因此 `frozen_backbone_verified_on_c16` 保持关闭。

完成标准：UniLION 进程退出、超时或输出异常时车辆必停；正常情况下特征能够稳定进入候选生成模块。

## 阶段 4：数据采集、训练与性能优化

0. `[已完成]` 建立项目内独立 Python 3.10 planner 环境，固定 CasADi 3.8.0；完成 0/8/16/32 个近场障碍的 11 次求解基准。四档均 11/11 `Solve_Succeeded`，热启动中位数分别为 1.92/13.87/23.00/40.53 ms，32 障碍热启动最大 40.65 ms，求解器单项满足 80 ms 时限。完整感知到控制链路仍需单独验收。
1. 采集仿真 rosbag、feature cache、MPC teacher label 和安全事件标签。
2. 训练候选头与安全头，执行消融实验并与四组基线比较。
3. 测量 C16/仿真点云到最终 `/cmd_vel` 的完整延迟分解。
4. 优化32个近场障碍的最坏情况，可选方案包括更严格的安全筛选、约束聚合、稀疏求解器或 acados；优化前继续使用超时停车策略。

完成标准：训练、验证、测试数据隔离，关键指标可复现，正常工况满足80 ms，超时和非有限输出始终安全停车。

## 阶段 5：C16 与 R680 实车验收

1. 上线 C16、底盘、里程计、IMU、TF、路径和急停，审计真实话题、频率、时间戳及 QoS。
2. 标定雷达到 `base_link` 外参，实测车体轮廓、速度/加速度、控制方向、通信延迟和制动距离。
3. 依次进行架空轮测试、空场零速/低速、静态障碍、动态障碍和复杂场景测试。
4. 用真实 C16 连续运行至少100帧 backbone 验证；完成前不解除 `frozen_backbone_verified_on_c16`。
5. 按证据逐项解锁23个 validation gates，并保留一键急停、人工接管和日志回放能力。

完成标准：所有实车安全门都有可追溯证据，端到端时限满足要求，任何单点故障均不能绕过安全监督输出危险命令。
