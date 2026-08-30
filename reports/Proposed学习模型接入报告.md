# Proposed 学习模型接入报告

日期：2026-08-30  
范围：开发机 ROS 2 Humble + Gazebo Classic，`empty/easy` 正向闭环及故障安全门验证。

## 1. 完成结论

阶段 F 主模型已经接入 Proposed 规划闭环。正常运行时控制器连续上报
`reason=learned_dcbf_mpc` 和 `learned_checkpoint_active=true`，不再使用预测裕量回退模式。

在线正向审计共收到 76 个控制器状态和 76 个 `/cmd_vel` 样本：

- 76/76 个状态均为 `learned_dcbf_mpc`；
- 76/76 个状态均为 `learned_checkpoint_active=true`；
- 控制命令全部为有限值，最大绝对分量为 `0.00129122599`；
- 单次观测的网络推理约 `1.76 ms`，D-CBF/MPC 求解约 `5.38 ms`；
- 学习候选索引为 0，MPC 状态为 `Solve_Succeeded`。

证据：`reports/proposed_active_runtime_audit.json`。

## 2. 运行链路

```text
Gazebo 16线点云
  → 字段适配与点云文件桥
  → 独立CUDA进程运行Frozen UniLION
  → 版本化环形特征缓存 + SHA-256
  → 阶段F PlanningSafetyModel
  → 安全头过滤并排序7条学习候选
  → 候选控制rollout
  → 学习候选作为CasADi D-CBF-MPC参考与initial_guess
  → 有限值/总时限/可行性门
  → 确定性近障碍安全监督
  → watchdog新鲜度和超时门
  → /cmd_vel
```

实现位置：

- `src/r680_safety_planner/learned_runtime.py`：checkpoint契约、输入输出、时限与候选排序校验；
- `scripts/proposed_inference_worker.py`：模型推理、学习候选排序、D-CBF-MPC和初值接入；
- `scripts/live_unilion_feature_worker.py`：在线UniLION特征及环形缓存；
- `simulation/ros2_ws/src/r680_sim_bringup/r680_sim_bringup/baseline_controller.py`：ROS闭环、watchdog和最终安全监督；
- `simulation/ros2_ws/src/r680_sim_bringup/launch/nav2_sim.launch.py`：Proposed一键启动；
- `scripts/audit_proposed_runtime.py`：在线激活/安全停车审计。

## 3. 强制模型契约

启动时必须全部满足以下条件，否则 worker 不进入 ready，控制器发布零速并上报
`checkpoint_validation_failed`：

| 项目 | 已验证值 |
|---|---|
| checkpoint SHA-256 | `54cf9c73c73f3353199047746875637c48b97ac623d267ef776380744d7a7ee9` |
| checkpoint格式 | `format_version=2` |
| 模型配置 | `384通道 / 7候选 / 20控制间隔 / 5维车辆状态 / 128隐藏维` |
| 训练代码提交 | `f59c1e746f4ce68b87edd6ca3c61ff0c7b85a3d7` |
| 数据manifest SHA-256 | `c9cef0ebc1f6fc5d6499542e8293ed9166135077c654a73a534b6b7605bd4d7d` |
| 数据版本 SHA-256 | `06ced7cdb0c20fadbf400dc1c4966ea895dd5192c9242422fff35f99b222142c` |
| UniLION checkpoint SHA-256 | `9ba65cfead901ea08db3664108838d4c85bd6621592cc08cca0dfe61af4eb262` |

训练代码提交号同时与冻结数据版本中的 `git_revision` 交叉校验；数据版本绑定manifest和
UniLION哈希。运行时只接受主模型，不接受消融checkpoint，权重以 strict 模式加载。

## 4. 网络输出如何参与控制

模型接收 `[384,32,32]` UniLION特征、`[32,4]` 路线、5维车辆状态和
`[3,60,60]` 代价地图。输出必须满足固定形状且全部为有限值。

候选首先依据安全头的可行性、预测最小屏障值和风险判定形成优先集合，再按候选logit排序；
未进入优先集合的候选只作为次级尝试。每条学习控制序列均重新rollout，并同时作为MPC参考轨迹
和 `initial_guess`。只有D-CBF/MPC可行且“网络推理 + 求解”不超过80 ms的候选才能生成命令。
最终命令还需经过近障碍停止/减速监督、有限值检查、里程计/特征/推理watchdog；这些边界不能被
学习输出绕过。

## 5. 故障测试

离线CUDA验证对主checkpoint执行了真实前向，并逐项注入以下故障：

| 故障 | 结果 |
|---|---|
| 模型文件缺失 | `FileNotFoundError`，拒绝激活 |
| checkpoint哈希错误 | `ValueError`，拒绝激活 |
| 网络输出NaN | 输出有限值门拒绝 |
| 推理超过80 ms | `TimeoutError`，拒绝输出 |
| UniLION特征超过0.30 s | `TimeoutError`，拒绝输入 |

五项均记录为 `safe_stop=true`，见 `reports/proposed_checkpoint_validation.json`。

另外执行了真实ROS闭环特征故障注入：精确停止在线UniLION worker后等待watchdog触发，随后
收到56/56个 `feature_timeout_stop` 状态，`learned_checkpoint_active` 全部为false；同期101个
`/cmd_vel` 样本全部为零。证据：`reports/proposed_feature_timeout_runtime_audit.json`。

## 6. 测试结果与边界

- 本地学习运行时单元测试：7/7通过；
- 本地完整单元测试：全部通过，1项按环境条件跳过；
- 开发机主模型CUDA契约/故障门：通过；
- 开发机Proposed正向ROS闭环：通过；
- 开发机UniLION退出安全停车：通过。

本阶段证明“Proposed确实使用训练模型闭环控制”以及要求的失效安全行为。它不替代后续八场景、
三难度、多随机种子的四控制器正式公平评测；该批量评测是下一阶段工作。实车实验不在当前计划中。
