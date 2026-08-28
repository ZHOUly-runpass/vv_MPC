# R680 训练样本格式

格式版本：`1.0`。单个样本使用压缩 NPZ，数据集索引使用 JSONL manifest。所有浮点数组必须为有限值，载入时会重新计算载荷 SHA-256；版本、形状、枚举或哈希不匹配会立即拒绝样本。

## 输入字段

| 字段 | 推荐 dtype | 形状 | 含义 |
|---|---:|---:|---|
| `points` | float32 | `[N,6]` | `x,y,z,intensity,ring,time` |
| `features` | float32 | `[C,H,W]` | 冻结 LiDAR backbone 的 BEV 特征 |
| `route` | float32 | `[R,4]` | 路点 `x,y,yaw,target_speed` |
| `ego_state` | float32 | `[S]` | 与车辆模型一致的当前状态 |
| `costmap` | float32 | `[3,Hc,Wc]` | 静态、动态、未知/可通行性三通道 |
| `obstacle_states` | float32 | `[O,T,6]` | 障碍预测 `x,y,yaw,vx,vy,valid_probability` |
| `obstacle_lengths/widths` | float32 | `[O,T]` | 障碍包围盒尺寸 |
| `obstacle_covariance` | float32 | `[O,T,2,2]` | 平面位置协方差 |
| `obstacle_valid_mask` | bool | `[O,T]` | 预测时刻是否有效 |
| `candidate_states` | float32 | `[K,T,S]` | K 条候选状态轨迹 |
| `candidate_controls` | float32 | `[K,T-1,U]` | 候选控制序列 |
| `candidate_timestamps_s` | float32 | `[T]` | 严格递增的相对时间 |

## 正式时间网格与重采样规则

正式训练和 teacher 的唯一时间网格是：预测域 `2.0 s`、`dt=0.1 s`、20 个控制区间、21 个状态/障碍时刻。新录制数据必须直接产生这一网格。

历史候选若使用 `dt=0.2 s`，转换规则固定为 `zero_order_hold_controls_rerollout_states_linear_obstacles`：控制量按零阶保持重采样到 0.1 s；候选状态不做坐标插值，而是从同一初始状态用车辆模型和重采样控制重新 rollout；障碍位置、速度、尺寸与协方差线性插值，有效掩码采用相邻源时刻的保守逻辑与。该规则及目标 `dt` 会写入样本/teacher 元数据，禁止在同一批正式标签中混用时间网格。

teacher求解前还必须把每条候选控制从样本的当前 `ego_state` 重新 rollout，消除候选消息与同步里程计之间的初始状态偏差；元数据记录 `teacher_candidate_anchor=ego_state_rerollout`。若IPOPT达到迭代上限，只允许使用受控停车候选作为初值重试一次，并把每次迭代数、活跃障碍数、初值来源和底层返回状态写入 `teacher_solver_diagnostics`。固定初始状态已经无法在最大slack内满足约束时，求解前明确标为 `Infeasible_Initial_State`，不得归为数值失败。

## MPC teacher 字段

| 字段 | dtype | 形状 | 含义 |
|---|---:|---:|---|
| `teacher_outcome_codes` | int8 | `[K]` | 0 成功、1 不可行、2 数值失败、3 超时 |
| `teacher_feasible` | bool | `[K]` | 求解器可行标志 |
| `teacher_h_min` | float32 | `[K]` | 最小安全屏障值 |
| `teacher_slack_max` | float32 | `[K]` | 最大 D-CBF 松弛量 |
| `teacher_solve_time_ms` | float32 | `[K]` | 单候选求解耗时 |
| `teacher_states` | float32 | `[K,T,S]` | teacher 优化后的状态 |
| `teacher_controls` | float32 | `[K,T-1,U]` | teacher 优化后的控制 |
| `teacher_selected_index` | int64 | 标量 | 最终监督候选索引 |

未标注样本的 teacher 向量必须为空，`teacher_selected_index=-1`。四种 outcome 的判定顺序固定为：超时 → 数值失败 → 不可行 → 成功，避免失败类型被 `feasible=false` 覆盖。

## 元数据与审计

元数据至少包含非空 `sample_id`，以及四个小写 64 位 SHA-256：

- `source_sha256`：原始 rosbag/帧源；
- `config_sha256`：场景、车辆、MPC 和预处理配置；
- `checkpoint_sha256`：冻结 backbone 权重；
- `code_sha256`：生成样本的代码快照。

文件另存 `payload_sha256`，它覆盖 schema 版本、规范化元数据以及每个数组的字段名、dtype、shape 和字节内容。manifest 每行保存相对路径、载荷哈希、sample ID、场景、seed、难度和 split。

## 数据隔离规则

真实数据必须以完整 rosbag/run 为单位划分 train/val/test；同一连续序列的相邻帧不得跨 split。建议另留未参与调参的场景—seed 组合做最终测试。合成夹具必须标记 `synthetic_fixture=true`，不得用于报告算法精度。

## rosbag 转换流水线

闭环录制入口的最后一个参数选择 `dwb`、`mppi`、`vanilla_dcbf` 或 `proposed`：

```bash
bash simulation/scripts/record_scenario.sh "$PWD" crossing_pedestrian 60 41 hard mppi
```

录制内容包括原始传感器、路线、Nav2 costmap、控制器状态，以及：

- `/planning/candidates`
- `/planning/obstacle_predictions`
- `/planning/mpc_request`
- `/planning/mpc_result`

在线未运行 CasADi teacher 时，`/planning/mpc_result` 明确使用 `offline_teacher_pending`，正式 teacher 字段由离线求解器生成。

teacher 必须显式读取车辆配置。仿真使用 `configs/robot/r680_sim.yaml`，其中尺寸、包络半径、速度/加速度约束和 MPC 参数与仿真 URDF/controller 对齐；实车配置 `configs/robot/r680_c16.yaml` 在 R680 参数未确认时会被拒绝，不能退回硬编码默认值生成正式标签。

一键转换命令：

```bash
bash simulation/scripts/convert_rosbag_to_schema.sh \
  "$PWD" BAG_DIRECTORY OUTPUT_DIRECTORY 2 32
```

转换分三步：纯 Python `rosbags` 解包并同步 → 独立 UniLION CUDA 环境批量生成冻结特征 → 组装并重新校验 schema 1.0。默认按 2 Hz 取样，并将原始 `[384,180,180]` BEV 自适应平均池化为 `[384,32,32]`，降低数据集磁盘占用；池化尺寸和特征配置哈希写入元数据。

当前Nav2只发布融合后的costmap，转换器将其编码为“融合占用、预留动态层、未知区域”三通道，并写入 `costmap_encoding=nav2_aggregate_reserved_dynamic_unknown`。动态障碍监督来自独立的 `/planning/obstacle_predictions`，不把融合costmap误称为可精确拆分的静态/动态图层。

组装后的 `raw_manifest.jsonl` 可直接传给 `generate_teacher_labels.py`。使用以下命令执行完整哈希、结构和 teacher outcome 审计：

```bash
PYTHONPATH="$PWD/src" .tools/envs/planner/bin/python scripts/audit_training_manifest.py \
  --manifest DATA/manifest.jsonl --output reports/dataset_audit.json
```
