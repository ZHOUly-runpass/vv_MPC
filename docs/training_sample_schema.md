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
