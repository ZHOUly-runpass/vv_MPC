# 仿真版：冻结 LiDAR 通用主干 + 学习型多候选规划 + D-CBF-MPC 完整执行方案

> 文档用途：作为当前项目的**主执行文档**，可直接交给内网 AI / 开发人员按阶段实现。  
> 当前范围：**只做仿真算法研究，不做 R680/C16 实车适配与实车实验。**  
> 研究主线：利用公开大规模 LiDAR 数据训练/预训练得到的通用三维表征，在 ROS 仿真环境中通过轻量适配器、多候选轨迹生成、安全预测与排序，为 D-CBF-MPC 提供高质量 warm-start，并完成闭环动态避障。  
> 原始背景：由 `算法完整流程.md` 中“冻结 LiDAR 主干 + Candidate + Safety/Ranker + D-CBF-MPC”主线重构而来；R680、C16、C50C、物理急停、实车参数标定等内容全部移至未来 Sim-to-Real 阶段。

---

# 1. 当前项目最终目标

## 1.1 当前阶段只解决一个问题

在统一 ROS 仿真环境中实现：

```text
公开 LiDAR 数据集预训练/已有 checkpoint
                │
                ↓
        Frozen LiDAR Backbone
                │
                ↓
        通用 BEV / 3D Feature
                │
        ┌───────┴────────┐
        │                │
Simulation LiDAR     Route / Ego / Costmap
        │                │
        └───────┬────────┘
                ↓
        LiDARFeatureAdapter
                ↓
      MultiCandidatePlanner
                ↓
     Candidate Pre-Filter
                ↓
     SafetyPredictionHeads
                ↓
        CandidateRanker
                ↓
              Top-K
                ↓
           D-CBF-MPC
                ↓
          Safe cmd_vel
                ↓
        Simulated Robot
```

研究目标不是让神经网络直接控制机器人，而是：

```text
Learning：缩小轨迹搜索空间，提供高质量候选 / warm-start
D-CBF-MPC：根据机器人模型和障碍物状态做在线约束优化
CBF：提供显式安全约束
ROS Simulator：形成完整闭环
```

---

# 2. 明确删除或延后的工作

当前阶段**不实现、不验收、不阻塞主线**的内容：

- WHEELTEC R680 实车型号确认；
- C50C 控制器；
- C16 精确子型号、真实驱动、真实外参；
- 实车急停；
- 实车制动距离；
- 电池、电机、串口/CAN；
- 实车 `car_mode`；
- 实车 command watchdog；
- 实车网络延迟；
- 实车方向符号验证；
- 物理 footprint 标定；
- 真实 IMU/odom 标定；
- 实车故障注入；
- 实车 motion unlock。

这些内容统一移动到：

```text
Future Work
└── Sim-to-Real / R680-C16 Deployment
```

当前所有配置、代码命名应尽量去除 `R680`、`C16` 强绑定。

例如：

```text
C16FeatureAdapter     -> LidarFeatureAdapter
R680VehicleModel      -> DifferentialDriveModel
r680_c16.yaml         -> simulation_diffdrive.yaml
```

---

# 3. 当前研究假设

整个项目需要验证以下核心假设：

## H1：通用 LiDAR 表征可迁移

使用 nuScenes / Waymo / KITTI / SemanticKITTI 等公开数据训练得到的 LiDAR backbone，即使数据来源是自动驾驶车辆，其**中低层三维几何特征**仍可为移动机器人导航提供有效场景表征。

注意：

```text
允许迁移：
Voxel / Pillar Encoder
Sparse 3D Backbone
BEV Backbone
部分 Temporal LiDAR Encoder

不直接迁移：
自动驾驶车辆 planning head
车辆轨迹监督
车道语义规划头
原车端控制输出
```

## H2：Learning Candidate 比盲目优化初始化更有效

学习网络能够根据 LiDAR、路径和自车状态，快速生成若干有明显行为差异的运动学可行候选：

```text
直行
左绕
右绕
减速
停车
必要时原地旋转
```

这些候选作为 D-CBF-MPC 的 warm-start，可减少：

- 不合理初值；
- 局部最优；
- 求解失败；
- 大幅修正；
- 在线搜索开销。

## H3：Learning 不替代 D-CBF

Safety Head / Ranker 只预测：

```text
feasible probability
predicted h_min
predicted slack
predicted correction
predicted risk
```

最终在线安全性仍由：

```text
真实几何障碍物状态
+ 机器人运动学
+ D-CBF 约束
+ MPC 状态/控制约束
```

决定。

---

# 4. 仿真平台选择

## 4.1 主推荐

### 新建环境

推荐：

```text
Ubuntu 24.04
ROS 2 Jazzy
Gazebo Harmonic / Modern Gazebo
Nav2
PyTorch
CasADi + IPOPT 或 acados
```

理由：

- ROS 2 Jazzy 与现代 Gazebo 是当前 Nav2 官方主路径；
- ROS/Gazebo bridge 完整；
- 可直接提供 odom、TF、IMU、LaserScan/PointCloud；
- Nav2 可作为 global planner 与 baseline 框架；
- DWB、MPPI baseline 可直接接入统一 Controller Server。

## 4.2 已有工程兼容路线

如果现有项目已经稳定运行：

```text
Ubuntu 22.04
ROS 2 Humble
Gazebo Classic
```

则**不要求为了本项目立刻迁移**。

原则：

```text
算法主线 > 中间件升级
```

只要以下接口满足，项目就可以继续：

```text
/clock
/tf
/odom
/scan 或 /points
/plan
/cmd_vel
```

## 4.3 暂不推荐作为主环境

### CARLA

当前不是首选，因为本项目研究对象是低速移动机器人自由空间局部导航，不是道路车辆规划。

### Isaac Sim

暂时作为第二仿真环境储备。

当后续需要研究：

```text
更真实的 3D LiDAR
复杂遮挡
高保真点云
域随机化
Sim-to-Real
```

再增加 Isaac Sim，不作为第一阶段 blocker。

---

# 5. 仿真机器人统一定义

为了让 DWB / MPPI / D-CBF-MPC / Proposed Method 公平比较，第一版只选择**差速机器人**。

## 5.1 状态

```math
x = [p_x, p_y, \theta, v, \omega]
```

最小实现也可使用：

```math
x = [p_x, p_y, \theta]
```

并将控制定义为：

```math
u = [v, \omega]
```

推荐最终 MPC 内部使用：

```math
u = [a_v, a_\omega]
```

从而显式控制速度变化率。

## 5.2 离散运动学

```math
p_{x,k+1} = p_{x,k} + v_k \cos(\theta_k) \Delta t
```

```math
p_{y,k+1} = p_{y,k} + v_k \sin(\theta_k) \Delta t
```

```math
\theta_{k+1} = \theta_k + \omega_k \Delta t
```

如果使用加速度控制：

```math
v_{k+1} = v_k + a_{v,k}\Delta t
```

```math
\omega_{k+1} = \omega_k + a_{\omega,k}\Delta t
```

## 5.3 第一版固定参数

不要使用实车参数。

使用仿真机器人参数：

```yaml
robot:
  model: diff_drive
  radius_m: 0.22
  v_max_mps: 0.6
  v_min_mps: -0.1
  omega_max_radps: 1.5
  a_v_max_mps2: 1.0
  a_omega_max_radps2: 2.5
```

后续所有算法共用这一组参数。

---

# 6. 传感器方案

## 6.1 必须保留两种感知输入

### A. 2D 几何安全通路

用于 D-CBF 的确定性障碍物输入：

```text
/scan
  ↓
局部聚类 / occupancy
  ↓
ObstacleState
  ↓
D-CBF-MPC
```

### B. 3D LiDAR 学习通路

用于 Frozen LiDAR Backbone：

```text
/points
  ↓
PointCloud2
  ↓
Voxelization
  ↓
Frozen Backbone
  ↓
BEV Feature
```

这是重要设计：

> 不要为了使用 3D backbone，让 D-CBF 也依赖深度网络检测结果。

D-CBF 第一版使用 simulator ground-truth / 几何聚类构造障碍物状态，保证安全模块和感知迁移问题解耦。

## 6.2 第一阶段障碍物真值

优先使用仿真器直接获取障碍物 pose / velocity：

```text
Gazebo model states
        ↓
GroundTruthObstacleBridge
        ↓
PredictedObstacle
```

目的：

先验证规划算法。

不要在第一阶段同时解决：

```text
3D detection
tracking
prediction
planning
control
```

否则无法判断失败来自哪一层。

## 6.3 第二阶段再加入感知噪声

完成 planner 后，再逐步替换：

```text
GT obstacle
  ↓
noisy GT
  ↓
geometry detection
  ↓
learned detection + tracker
```

---

# 7. 场景设计

至少实现以下 8 类 world。

```text
world_01_empty
world_02_static_sparse
world_03_static_dense
world_04_narrow_passage
world_05_crossing_pedestrian
world_06_head_on_dynamic
world_07_multi_dynamic
world_08_local_minimum_trap
```

## 7.1 Empty

用途：

- 检查路径跟踪；
- 检查控制器稳定性；
- 检查模型无障碍物时是否过度绕行。

## 7.2 Static Sparse

少量静态圆柱 / box。

用途：基础绕障。

## 7.3 Static Dense

随机布置多个障碍物。

用途：复杂局部规划。

## 7.4 Narrow Passage

设置略大于机器人直径的通道。

用途：

- D-CBF 可行性；
- 过度保守问题；
- ranker 是否倾向停车。

## 7.5 Crossing Pedestrian

动态障碍横穿 global path。

用途：动态避障核心场景。

## 7.6 Head-on

障碍物沿路径反方向靠近。

用途：TTC 与制动/绕行行为。

## 7.7 Multi Dynamic

2~5 个动态障碍。

用途：多障碍约束和计算负载。

## 7.8 Local Minimum Trap

构造 U 型 / 半封闭场景。

用途：检验候选多样性是否帮助 MPC 避免局部最优。

---

# 8. ROS 软件架构

推荐工作区：

```text
e2empc_ws/
├── src/
│   ├── sim_robot_description/
│   ├── sim_worlds/
│   ├── sim_bringup/
│   ├── lidar_preprocess/
│   ├── lidar_backbone_bridge/
│   ├── lidar_feature_adapter/
│   ├── route_encoder/
│   ├── ego_encoder/
│   ├── candidate_planner/
│   ├── candidate_filter/
│   ├── safety_prediction/
│   ├── candidate_ranker/
│   ├── obstacle_interface/
│   ├── obstacle_prediction/
│   ├── dcbf_mpc/
│   ├── proposed_controller/
│   ├── baseline_dwb/
│   ├── baseline_mppi/
│   ├── benchmark_manager/
│   ├── dataset_recorder/
│   └── evaluation_tools/
│
├── configs/
│   ├── simulation_diffdrive.yaml
│   ├── backbone_centerpoint.yaml
│   ├── backbone_unilion.yaml
│   ├── planner.yaml
│   ├── dcbf_mpc.yaml
│   └── benchmark.yaml
│
├── training/
│   ├── datasets/
│   ├── models/
│   ├── losses/
│   ├── teachers/
│   ├── scripts/
│   └── checkpoints/
│
├── data/
│   ├── raw_rosbag/
│   ├── processed/
│   ├── frozen_features/
│   ├── teacher_labels/
│   └── benchmark_results/
│
└── docs/
    ├── interfaces.md
    ├── training.md
    ├── benchmark.md
    └── debugging.md
```

---

# 9. ROS Topic 契约

统一使用以下核心接口：

```text
/clock
/tf
/tf_static
/odom
/scan
/points
/plan
/cmd_vel
```

项目内部新增：

```text
/perception/bev_feature
/perception/obstacles
/planner/candidates
/planner/filtered_candidates
/planner/ranked_candidates
/mpc/safe_trajectory
/mpc/debug
/controller/debug
```

不要让 Python 模型直接依赖 Nav2 私有内部数据结构。

所有模块通过稳定 message / tensor schema 交互。

---

# 10. Frozen LiDAR Backbone 路线

## 10.1 第一版工程基线

优先：

```text
CenterPoint / OpenPCDet
```

原因：

- 开源成熟；
- 有 nuScenes checkpoint；
- BEV backbone 明确；
- 容易提取中间 BEV feature；
- 依赖结构比大型统一模型简单；
- 适合作为 frozen backbone 原型。

## 10.2 第二版研究路线

再增加：

```text
UniLION LiDAR-only
或
UniLION temporal LiDAR
```

不要一开始同时开发两套 backbone。

执行顺序：

```text
CenterPoint frozen backbone
        ↓
完整 pipeline 跑通
        ↓
定义统一 FrozenSceneFeatures
        ↓
UniLION adapter
```

---

# 11. 自动驾驶公开数据集的使用边界

## 11.1 可以用于

```text
LiDAR backbone 预训练
3D geometry representation
BEV representation
可选 temporal LiDAR representation
```

推荐优先级：

```text
1. 直接使用官方/成熟开源 checkpoint
2. 不足时再自行预训练
```

当前阶段不建议从零训练大型 LiDAR backbone。

## 11.2 不可以直接用于

```text
小车 candidate trajectory supervision
小车控制策略
小车 D-CBF 标签
小车 route-following policy
```

原因：

自动驾驶车辆和差速移动机器人在：

```text
尺度
速度
运动学
自由空间
行为模式
障碍物距离
控制输出
```

上差异过大。

## 11.3 数据源最终划分

```text
Dataset A：公开 LiDAR 数据
用途：Backbone pretraining / checkpoint

Dataset B：ROS Simulation Dataset
用途：Adapter / Candidate / Safety / Ranker

Dataset C：D-CBF-MPC Teacher Label
用途：Safety Head / Ranker supervision
```

---

# 12. Frozen Feature 接口

所有 backbone 最终必须转换到统一契约：

```python
FrozenSceneFeatures = {
    "schema_version": "1.0",
    "bev_feature": Tensor[B, C, H, W],
    "bev_resolution_m": float,
    "bev_origin_xy": Tensor[2],
    "valid_mask": Optional[Tensor],
    "timestamp": float,
    "source_backend": str,
    "checkpoint_hash": str,
}
```

第一版只强制：

```text
bev_feature
bev_resolution
coordinate definition
```

不要把 detection head 输出作为必需字段。

---

# 13. Feature Adapter

目的：解决两类 domain gap：

```text
自动驾驶 LiDAR 域
        ↓
移动机器人仿真 LiDAR 域

大范围 BEV
        ↓
局部导航 BEV
```

输入：

```text
Frozen BEV Feature
Local Route
Ego State
可选 Costmap Feature
```

第一版网络建议：

```text
BEV Feature
   ↓
1x1 Conv / ConvBlock
   ↓
Channel reduction
   ↓
ROI crop / resample
   ↓
BEV Adapter
   ↓
Navigation Feature
```

不要第一版就使用复杂 Transformer。

输出建议：

```text
[B, 128, Hn, Wn]
```

例如局部范围：

```text
x ∈ [-2, 8] m
y ∈ [-5, 5] m
```

以机器人为中心。

---

# 14. Route Encoder

输入：Nav2 global path 在机器人局部坐标系中的采样点。

固定采样：

```python
route_points: [B, Nr, 4]
# [x, y, sin(yaw), cos(yaw)]
```

推荐：

```text
Nr = 20~40
```

第一版：

```text
MLP
↓
1D Conv
↓
Route Token / Route Feature
```

不要依赖 HD Map。

---

# 15. Ego Encoder

输入：

```python
[v, omega]
```

可增加最近 5~10 帧：

```python
[v_t, omega_t]
```

第一版使用 MLP 即可。

---

# 16. MultiCandidatePlanner

这是核心学习模块之一。

## 16.1 不直接回归 XY 轨迹

优先预测**控制序列 / control knots**：

```python
candidate_controls:
[B, G, Nc, 2]
```

其中：

```text
G  = candidate 数量
Nc = control knots 数量
2  = [v, omega] 或 [a_v, a_omega]
```

再通过 Differentiable Kinematic Rollout：

```text
control knots
     ↓
interpolation
     ↓
robot model rollout
     ↓
trajectory states
```

好处：

- 天然满足运动学；
- Candidate 与 MPC 接口统一；
- 不会出现 XY 插值后控制不可实现的问题。

## 16.2 推荐第一版参数

```yaml
planning:
  horizon_s: 2.0
  candidate_dt_s: 0.2
  candidate_count: 8
  control_knots: 10
  top_k_to_mpc: 3
```

8 个候选不是 8 个固定类别，而是 8 个 mode。

网络必须通过 diversity loss 防止 collapse。

## 16.3 必须保留 Stop Candidate

无论网络输出如何，系统始终追加一个解析候选：

```text
STOP
```

作为安全 fallback。

---

# 17. Candidate 时间接口统一

必须解决原方案中 Candidate `0.2 s` 与 MPC `0.1 s` 的时间尺度不一致。

规范：

```text
Candidate control knots：dt = 0.2 s
MPC internal step：dt = 0.1 s
```

转换流程必须是：

```text
Candidate control knots
        ↓
控制输入插值
        ↓
MPC dt = 0.1 s
        ↓
用 robot dynamics 重新 rollout
        ↓
MPC warm-start state/control sequence
```

禁止：

```text
直接对 candidate XY trajectory 线性插值
```

因为可能破坏动力学一致性。

---

# 18. Candidate Pre-Filter

进入 Ranker / MPC 前先做解析检查。

每个候选计算：

```text
speed_max
omega_max
acceleration_max
route_deviation
collision_min_distance
TTC
progress
trajectory_length
costmap_collision
```

硬约束不满足：

```text
valid = false
```

不要交给神经网络判断明显非法轨迹。

---

# 19. 障碍物状态接口

统一定义：

```python
PredictedObstacle = {
    "id": int,
    "states": Tensor[T, 4],
    # x, y, vx, vy
    "radius": float,
    "covariance": Tensor[T, 2, 2],
    "source": str,
}
```

第一版：

```text
source = simulator_ground_truth
covariance = small fixed value
```

第二版：

增加噪声和 tracker。

---

# 20. Dynamic Obstacle Prediction

第一版只实现 CV：

```math
p_{k+1} = p_k + v_k \Delta t
```

即 Constant Velocity。

不要一开始训练深度运动预测网络。

后续扩展顺序：

```text
CV
↓
Kalman Filter
↓
Social / Learned Predictor
```

---

# 21. D-CBF Barrier Function

第一版机器人和障碍物都使用圆形近似。

机器人安全半径：

```math
r_s = r_{robot} + r_{obs} + m_{safe}
```

定义：

```math
h(x,o) = (p_x-o_x)^2 + (p_y-o_y)^2 - r_s^2
```

安全区域：

```math
h(x,o) \ge 0
```

这种定义的单位为：

```text
m²
```

因此所有：

```text
h_min
slack
barrier threshold
```

都必须统一按照 `m²` 解释。

禁止同时混用：

```text
distance margin (m)
和
squared-distance barrier (m²)
```

---

# 22. 离散 D-CBF 约束

采用：

```math
h_{k+1} \ge (1-\gamma)h_k - \epsilon_k
```

其中：

```math
\gamma(\Delta t)=1-\exp(-\alpha \Delta t)
```

```math
\epsilon_k \ge 0
```

第一版必须记录：

```text
h_min
slack_max
slack_sum
active_constraint_count
```

---

# 23. D-CBF-MPC 优化问题

## 23.1 决策变量

```text
X = x_0 ... x_N
U = u_0 ... u_{N-1}
Slack = ε_0 ... ε_{N-1}
```

## 23.2 目标函数

```math
J =
\sum_k
w_p ||p_k-p_k^{ref}||^2
+
w_\theta e_{\theta,k}^2
+
w_u ||u_k||^2
+
w_{du} ||u_k-u_{k-1}||^2
+
w_{cand} ||x_k-x_k^{cand}||^2
+
w_s \epsilon_k^2
```

其中：

```text
reference path：来自 Nav2 global path
candidate：来自 neural candidate
```

这非常重要：

Candidate 不作为硬约束，而作为：

```text
warm start
+
soft reference
```

因此 MPC 仍有能力修正候选。

## 23.3 约束

```text
robot dynamics
v_min <= v <= v_max
|omega| <= omega_max
|a_v| <= a_v_max
|a_omega| <= a_omega_max
D-CBF constraints
slack >= 0
```

---

# 24. MPC 求解器

第一版推荐二选一：

## A. CasADi + IPOPT

优点：开发简单、调试方便。

用于：

```text
正确性优先
第一版算法验证
离线 Teacher
```

## B. acados

用于：

```text
后续实时优化
```

执行顺序：

```text
先 CasADi 跑通
↓
接口稳定
↓
需要实时性时再迁移 acados
```

不要第一阶段被求解器工程复杂度拖慢。

---

# 25. Top-K D-CBF-MPC 流程

```text
Ranked Candidate 1
      ↓
MPC solve
 ├─ feasible → 输出
 └─ fail
      ↓
Ranked Candidate 2
      ↓
MPC solve
 ├─ feasible → 输出
 └─ fail
      ↓
Ranked Candidate 3
      ↓
MPC solve
 ├─ feasible → 输出
 └─ fail
      ↓
STOP Candidate
```

训练阶段可以对所有 G 个候选运行 teacher。

在线阶段只对 Top-K 求解。

---

# 26. D-CBF-MPC Teacher 数据生成

这是训练 Safety Head / Ranker 的关键。

对于每个 simulation sample：

```text
scene feature
route
state
obstacles
candidate_1 ... candidate_G
```

对每条 candidate 独立运行 D-CBF-MPC。

保存：

```python
TeacherLabel = {
    "candidate_id": int,
    "feasible": bool,
    "solve_status": str,
    "h_min": float,
    "slack_max": float,
    "slack_sum": float,
    "correction_norm": float,
    "objective": float,
    "solve_time_ms": float,
    "progress": float,
    "collision": bool,
}
```

必须区分：

```text
INFEASIBLE
SOLVER_TIMEOUT
NUMERICAL_FAILURE
SUCCESS
```

不能全部合并成 `feasible=false`。

---

# 27. SafetyPredictionHeads

输入：

```text
Scene Feature
+ Candidate Feature
+ Obstacle Context
```

输出：

```python
{
    "feasibility_logit": ..., 
    "predicted_h_min": ...,
    "predicted_slack": ...,
    "predicted_correction": ...,
    "predicted_mpc_cost": ...,
}
```

第一版不强制单独预测 `risk`。

Risk 可从这些可解释量组合得到。

---

# 28. CandidateRanker

Ranker 预测每条 candidate 的最终排序分数。

建议分数语义：

```text
高分 =
高 MPC 可行概率
+ 高 progress
+ 低 correction
+ 高 h_min
+ 低 slack
+ 低控制代价
```

Ranker 不直接学习碰撞标签即可结束。

它需要学习：

> 哪一条 candidate 最值得优先交给昂贵的 MPC。

---

# 29. 训练数据采集

ROS 仿真中自动生成 episode。

每个 episode 随机化：

```text
start pose
goal pose
static obstacle count
static obstacle pose
dynamic obstacle count
dynamic speed
dynamic direction
sensor noise
robot initial velocity
```

每个规划时刻保存：

```python
SimulationSample = {
    "timestamp": ...,
    "points": ...,
    "odom": ...,
    "local_route": ...,
    "obstacles_gt": ...,
    "world_id": ...,
    "episode_id": ...,
}
```

---

# 30. 数据缓存策略

重型 backbone 不应该在每次 head 训练时重复运行。

流程：

```text
rosbag / simulation samples
        ↓
Frozen Backbone
        ↓
cache BEV feature
        ↓
training dataset
```

缓存：

```python
{
  "bev": float16 tensor,
  "ego": ...,
  "route": ...,
  "obstacles": ...,
  "metadata": ...
}
```

metadata 必须含：

```text
backbone name
checkpoint hash
voxel config
BEV resolution
coordinate convention
```

---

# 31. Candidate Planner 监督来源

推荐按以下顺序构造。

## 31.1 Teacher MPC / D-CBF-MPC

对当前状态使用多个不同参考 / 初始化运行优化器，生成不同 mode 的优质轨迹。

例如：

```text
center
left_bias
right_bias
slow
stop
```

## 31.2 Nav2 MPPI 轨迹

可作为额外 imitation source。

## 31.3 规则候选

用于 bootstrap：

```text
constant v/w
left arc
right arc
slowdown
stop
```

第一版不要求 candidate head 一开始完全由专家轨迹学习。

---

# 32. Candidate Planner Loss

建议：

```math
L_{plan} =
L_{best-of-G}
+ \lambda_{ctrl}L_{control}
+ \lambda_{route}L_{route}
+ \lambda_{div}L_{diversity}
+ \lambda_{dyn}L_{dynamics}
```

其中最关键：

## Best-of-G

只要求至少一个 candidate 接近 teacher，避免所有 mode 被迫平均。

## Diversity Loss

防止 8 条 candidate 全部重叠。

可基于：

```text
trajectory endpoint distance
control sequence distance
pairwise trajectory distance
```

---

# 33. Safety Head Loss

```math
L_{safe} =
\lambda_f L_{BCE}(feasible)
+
\lambda_h L_1(\hat h_{min}, h_{min})
+
\lambda_s L_1(\hat s, s)
+
\lambda_c L_1(\hat c, c)
```

对 solver timeout / numerical failure 使用 mask 或单独 class。

不要把 numerical failure 当作真实 unsafe 样本。

---

# 34. Ranker Loss

推荐 pairwise ranking：

对于同一 scene：

```text
candidate i 的 teacher utility > candidate j
```

则要求：

```math
score_i > score_j
```

Teacher utility 可定义：

```math
U =
w_f I(feasible)
+w_p progress
+w_h normalize(h_min)
-w_c correction
-w_s slack
-w_j objective
```

第一版 utility 权重写入配置文件，不能硬编码。

---

# 35. 完整训练阶段

## Stage 0：仿真闭环基础

必须完成：

```text
Gazebo world
robot
odom
TF
LiDAR
Nav2 global path
cmd_vel
```

验收：

- teleop 正常；
- Nav2 DWB 能从 start 到 goal；
- rosbag 可记录；
- 动态障碍脚本正常。

## Stage 1：D-CBF-MPC 独立 baseline

输入：

```text
robot state
route
GT obstacles
```

不使用神经网络。

验收：

- Empty 可跟踪路径；
- Static 可绕障；
- Crossing 可停车/绕行；
- `h_min >= -tolerance`；
- solver 状态可记录。

## Stage 2：Frozen Backbone

先接 CenterPoint。

验收：

```text
PointCloud2 -> voxel -> backbone -> BEV
```

连续数据：

- 无 NaN/Inf；
- shape 稳定；
- timestamp 对齐；
- feature variance 非零。

注意：

这一阶段不要求原 CenterPoint detection head 在机器人环境中检测准确。

## Stage 3：Feature Cache

批量生成 frozen features。

验收：

- cache 可重复加载；
- checkpoint hash 正确；
- 同一 sample 特征一致。

## Stage 4：Candidate Planner

训练：

```text
Adapter
RouteEncoder
EgoEncoder
MultiCandidatePlanner
```

验收：

- candidate 不 collapse；
- 满足控制边界；
- 至少能产生左/右/直/慢等不同 mode。

## Stage 5：Teacher Label

对所有候选运行 D-CBF-MPC。

验收：

- 每条 candidate 有完整 teacher label；
- timeout / failure / infeasible 分开；
- 数据可重放。

## Stage 6：Safety Head

冻结 Candidate Planner，训练 Safety Heads。

验收：

- feasibility 分类可用；
- h_min / correction / slack 有基本相关性。

## Stage 7：Ranker

训练候选排序。

验收：

Top-1 / Top-K 命中 teacher 最优或可行 candidate 的比例明显高于随机排序。

## Stage 8：完整闭环

```text
LiDAR
→ Frozen Backbone
→ Adapter
→ Candidate
→ Safety
→ Ranker
→ Top-K D-CBF-MPC
→ cmd_vel
```

实现 ROS controller node。

## Stage 9：可选联合微调

只有 Stage 8 稳定以后才做：

```text
Adapter + Candidate + Safety + Ranker
```

小学习率联合训练。

第一版不要解冻 backbone。

---

# 36. Baseline

当前执行阶段至少实现 4 个控制器。

## B0：DWB

Nav2 DWB。

作用：经典 Dynamic Window / trajectory critic baseline。

## B1：MPPI

Nav2 MPPI。

作用：sampling-based predictive control baseline。

## B2：Vanilla D-CBF-MPC

```text
route reference
+ D-CBF-MPC
```

不使用 learned candidate。

初始化使用：

```text
previous solution
或
constant-speed rollout
```

这是最重要的算法 baseline。

## B3：Proposed

```text
Frozen LiDAR
+ Adapter
+ MultiCandidate
+ Safety Head
+ Ranker
+ D-CBF-MPC
```

可选后续增加：

```text
B4：Candidate + MPC without CBF
B5：Candidate + CBF-MPC without Ranker
```

但不是第一阶段 blocker。

---

# 37. 当前不以“实验论文结果”为工作目标，但代码必须保留评估接口

虽然当前不需要开展正式实验，仍需从第一天记录：

```text
success
collision
minimum_distance
h_min
slack
solve_time
planning_time
path_length
travel_time
control_smoothness
candidate_filter_rate
MPC correction_norm
solver_failures
```

原因：

后续其他人员接手实验时，不需要重新修改主代码。

---

# 38. 统一 Benchmark Manager

实现：

```text
benchmark_manager
```

负责：

```text
加载 world
设置 random seed
设置 start/goal
生成动态障碍
选择 controller
启动 episode
判断 success/failure
保存日志
reset world
```

控制器通过参数切换：

```yaml
controller: dwb
# mppi
# dcbf_mpc
# proposed
```

禁止为不同算法建立不同 world / robot 配置。

---

# 39. 配置文件重构

新增：

```text
configs/simulation_diffdrive.yaml
```

建议结构：

```yaml
simulation:
  ros_distro: jazzy
  simulator: gazebo_harmonic
  use_sim_time: true

robot:
  model: differential_drive
  radius_m: 0.22
  v_max_mps: 0.6
  omega_max_radps: 1.5
  a_v_max_mps2: 1.0
  a_omega_max_radps2: 2.5

lidar:
  type: simulated_3d
  topic: /points
  frame: lidar_link
  roi:
    x: [-10.0, 15.0]
    y: [-10.0, 10.0]
    z: [-2.0, 3.0]

backbone:
  primary: centerpoint
  frozen: true
  checkpoint: null
  feature_key: spatial_features_2d

planning:
  horizon_s: 2.0
  candidate_dt_s: 0.2
  candidate_count: 8
  top_k: 3

mpc:
  dt_s: 0.1
  horizon_s: 2.0
  intervals: 20
  solver: ipopt

cbf:
  barrier: squared_circle_distance
  alpha: 2.0
  safety_margin_m: 0.15
  allow_slack: true

benchmark:
  seed: 42
  worlds:
    - empty
    - static_sparse
    - static_dense
    - narrow
    - crossing
    - head_on
    - multi_dynamic
    - local_minimum
```

原 `r680_c16.yaml` 保留为：

```text
future_real_robot_config
```

不要继续作为当前主配置。

---

# 40. 模块接口文件

必须第一时间编写：

```text
docs/interfaces.md
```

内容至少定义：

```text
Coordinate frame
Timestamp
Point cloud schema
BEV coordinate
Route schema
Robot state
Candidate control
Candidate state
Obstacle state
Teacher label
MPC input/output
```

任何模块实现前先遵守这个接口。

这是避免项目后期大量返工的关键。

---

# 41. 推荐坐标约定

统一：

```text
robot local frame
x forward
y left
z up
yaw counter-clockwise
```

所有 neural planning target 使用 robot-local frame。

优点：

- 与 global map 解耦；
- 数据增强简单；
- 不同 world 可共享；
- 不需要网络学习绝对地图坐标。

---

# 42. Debug 可视化

RViz 必须提供：

```text
Global Path
Robot footprint
LiDAR
Obstacles
All Candidates
Filtered Candidates
Ranked Top-K
MPC Safe Trajectory
CBF Active Obstacles
```

不同 candidate marker 通过不同 namespace 区分。

必须可以点击/日志查看：

```text
rank score
predicted feasible
predicted h_min
teacher / actual MPC h_min
MPC correction
```

---

# 43. 单元测试

至少实现：

## dynamics

```text
zero input
straight
constant yaw rate
rotate in place
```

## candidate rollout

```text
candidate dt -> MPC dt resample
```

## barrier

```text
outside safety radius => h > 0
on boundary => h = 0
inside => h < 0
```

## MPC

```text
no obstacle
static obstacle
dynamic obstacle
infeasible initial state
```

## coordinate

```text
global path -> robot local
local -> global round trip
```

---

# 44. 不允许的实现捷径

内网 AI 实现时禁止以下做法：

1. **禁止**直接使用自动驾驶 planning head 控制小车。
2. **禁止**把 nuScenes trajectory 当作小车控制监督。
3. **禁止**让 Safety Head 代替 D-CBF 约束。
4. **禁止**只用检测类别作为障碍物来源。
5. **禁止**直接对 XY candidate 线性插值作为 MPC warm-start。
6. **禁止**Candidate 与 MPC 使用不同坐标系但无显式转换。
7. **禁止**把 solver timeout 标成 obstacle infeasible。
8. **禁止**训练 backbone 后再随意改变 voxel range / feature schema。
9. **禁止**所有 candidate collapse 成相同轨迹而仍继续训练 Safety Head。
10. **禁止**一开始同时上 CenterPoint、UniLION、learned prediction、Isaac Sim、实车。

---

# 45. 推荐开发顺序（真正执行顺序）

严格按照：

```text
P0  ROS Simulation
│
├─ P1  Nav2 + DWB 跑通
│
├─ P2  Ground-truth obstacle interface
│
├─ P3  Vanilla MPC
│
├─ P4  D-CBF-MPC
│
├─ P5  Dynamic obstacle benchmark
│
├─ P6  CenterPoint Frozen Backbone
│
├─ P7  Frozen Feature Cache
│
├─ P8  Rule-based Multi-Candidate Generator
│
├─ P9  Candidate -> D-CBF-MPC Top-K interface
│
├─ P10 Learned Candidate Planner
│
├─ P11 Teacher Label Generator
│
├─ P12 Safety Prediction Head
│
├─ P13 Candidate Ranker
│
├─ P14 Complete Proposed Controller
│
├─ P15 MPPI Baseline
│
└─ P16 Optional UniLION Backbone
```

为什么在 learned candidate 前先做 rule-based candidate：

> 为了先验证 “Multi-Candidate → Top-K → D-CBF-MPC” 软件接口是否正确，而不是一开始让训练问题阻塞系统联调。

---

# 46. 每个阶段必须交付的文件

## P0-P2

```text
sim_bringup/
sim_worlds/
robot_description/
obstacle_interface/
```

## P3-P4

```text
dcbf_mpc/
  dynamics.py
  barrier.py
  solver.py
  controller_node.py
  tests/
```

## P6-P7

```text
lidar_backbone_bridge/
feature_cache/
```

## P8-P10

```text
candidate_planner/
  rule_generator.py
  model.py
  rollout.py
  losses.py
```

## P11-P13

```text
teachers/dcbf_teacher.py
safety_prediction/model.py
candidate_ranker/model.py
```

## P14

```text
proposed_controller/
```

统一对 ROS 输出：

```text
/cmd_vel
```

---

# 47. 每阶段验收 Gate

## Gate A：Simulation Ready

```text
[ ] robot spawn
[ ] TF correct
[ ] odom correct
[ ] LiDAR publishes
[ ] cmd_vel works
[ ] Nav2 global path works
```

## Gate B：D-CBF Ready

```text
[ ] empty path tracking
[ ] static avoidance
[ ] dynamic avoidance
[ ] h calculation verified
[ ] slack logged
[ ] failure reason logged
```

## Gate C：Backbone Ready

```text
[ ] checkpoint loads
[ ] simulation point cloud accepted
[ ] BEV feature shape fixed
[ ] BEV feature no NaN/Inf
[ ] coordinate metadata defined
```

## Gate D：Candidate Ready

```text
[ ] G candidates generated
[ ] dynamics rollout valid
[ ] stop candidate exists
[ ] diversity checked
[ ] pre-filter works
```

## Gate E：Teacher Ready

```text
[ ] every candidate has MPC label
[ ] status categories separated
[ ] teacher results reproducible
```

## Gate F：Learning Ready

```text
[ ] Safety Head trains
[ ] Ranker trains
[ ] Top-K interface works
```

## Gate G：Closed-loop Ready

```text
[ ] Proposed controller runs continuously
[ ] MPC fail has fallback
[ ] all internal debug topics publish
[ ] benchmark manager can reset episode
```

---

# 48. 第一阶段最低可运行版本 MVP

MVP 不包含神经网络。

```text
Gazebo
+ DiffDrive Robot
+ 3D/2D LiDAR
+ Nav2 Global Path
+ GT Dynamic Obstacles
+ D-CBF-MPC
```

完成之后才能开始学习模块。

这是强制要求。

原因：

如果 D-CBF-MPC baseline 自身都不能稳定工作，则后面不能判断 learning module 是否真正带来收益。

---

# 49. 第二阶段 MVP

```text
MVP-1
+
Frozen CenterPoint BEV Feature
+
Rule-based Candidates
+
Top-K D-CBF-MPC
```

这里仍然不训练 Candidate Planner。

目的是验证：

```text
backbone runtime
candidate interface
MPC warm-start
ROS scheduling
```

---

# 50. 第三阶段 MVP

```text
Frozen CenterPoint
+
Learned Candidate Planner
+
D-CBF-MPC Teacher
+
Safety Head
+
Ranker
+
Top-K D-CBF-MPC
```

这是当前项目算法完整版本。

---

# 51. Future Work：Sim-to-Real

当前项目完成后，未来其他人员再做：

```text
Simulation LiDAR
        ↓
Real C16

Simulation DiffDrive
        ↓
R680 actual chassis

simulation timing
        ↓
real timestamp / latency

circle footprint
        ↓
measured footprint

GT obstacle
        ↓
geometry / detector / tracker
```

届时再重新启用原始文档中的：

```text
R680 commissioning
C16 field validation
TF calibration
command watchdog
physical emergency stop
braking calibration
motion unlock
```

当前代码需要为此预留接口，但不需要实现。

---

# 52. 给内网 AI 的最终执行指令

内网 AI 应按照以下原则执行：

1. 先读取本文件，不以旧的 R680 实车流程作为当前主线。
2. 当前只创建仿真相关代码和配置。
3. 第一机器人类型固定为差速机器人。
4. 首先保证 ROS + Gazebo + Nav2 + D-CBF-MPC baseline 成功。
5. Frozen Backbone 第一版使用 CenterPoint/OpenPCDet。
6. Backbone 只负责特征提取，不依赖其检测 head 作为安全唯一输入。
7. 自动驾驶公开数据只用于 backbone 预训练/checkpoint。
8. Planning/Safety/Ranker 数据全部从 ROS simulation + D-CBF teacher 产生。
9. MultiCandidateHead 优先预测 control knots，而不是直接预测 XY。
10. Candidate 与 MPC 时间尺度通过“控制插值 + dynamics rollout”统一。
11. D-CBF 第一版使用圆形 squared-distance barrier，并明确单位为 `m²`。
12. MPC 第一版使用 CasADi + IPOPT，稳定后再考虑 acados。
13. 所有模块必须保存中间输出和 debug 信息。
14. 每完成一个 Gate 后再进入下一阶段。
15. 不允许因未来实车需求阻塞当前仿真算法开发。

---

# 53. 当前版本的一句话总结

```text
公共 LiDAR 数据负责“学会看三维世界”，
ROS 仿真数据负责“学会给小车提出走法”，
D-CBF-MPC 负责“判断并修正这些走法是否安全可执行”。
```

这三部分必须在数据来源、训练目标和在线职责上保持明确分离。

---

# 54. 推荐技术栈汇总

```text
OS:
  Ubuntu 24.04（新建）/ Ubuntu 22.04（已有 Humble 可保持）

ROS:
  ROS 2 Jazzy（新建推荐）
  ROS 2 Humble（现有项目兼容）

Simulator:
  Gazebo Harmonic（Jazzy）
  Gazebo Classic（Humble 现有工程）

Navigation:
  Nav2

Baselines:
  DWB
  MPPI
  Vanilla D-CBF-MPC

LiDAR Backbone:
  CenterPoint/OpenPCDet（第一版）
  UniLION LiDAR-only / temporal LiDAR（第二版）

Learning:
  PyTorch

MPC:
  CasADi + IPOPT（第一版）
  acados（可选实时优化）

Data:
  rosbag2
  numpy / torch tensor cache
  parquet/json/csv metadata

Visualization:
  RViz2
  TensorBoard
```

---

# 55. 开发完成定义（Definition of Done）

当前仿真项目只有在以下条件全部满足后才算“算法流程实现完成”：

```text
[ ] ROS 仿真机器人可重复启动
[ ] Nav2 global path 正常
[ ] 动态障碍物 benchmark 可自动生成
[ ] Vanilla D-CBF-MPC 可独立闭环
[ ] Frozen CenterPoint 可输出稳定 BEV feature
[ ] Feature cache 可离线训练复用
[ ] MultiCandidatePlanner 可输出运动学一致的多候选
[ ] Candidate pre-filter 可工作
[ ] D-CBF Teacher 可为每条候选生成结构化标签
[ ] SafetyPredictionHeads 完成训练接口
[ ] CandidateRanker 完成训练接口
[ ] Top-K Candidate 可送入 D-CBF-MPC
[ ] Proposed Controller 可持续闭环输出 cmd_vel
[ ] DWB / MPPI / Vanilla D-CBF-MPC 与 Proposed 使用同一仿真接口
[ ] Benchmark Manager 可统一启动、reset、记录
[ ] 所有中间结果可在 RViz / 日志中调试
[ ] 原 R680/C16 实车配置不再阻塞当前仿真主线
```

当这些条件满足后，后续实验人员可以直接在统一 benchmark 上开展参数、消融和算法对比，而无需重新修改系统架构。
