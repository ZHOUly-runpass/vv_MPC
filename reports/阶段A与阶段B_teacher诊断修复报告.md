# 阶段A与阶段B：MPC teacher诊断与修复报告

日期：2026-08-28  
开发机：Ubuntu 22.04 / ROS 2 Humble / CasADi 3.8 / IPOPT

## 验收结论

阶段A和阶段B已完成。静态场景专用复验矩阵达到35/35有效样本、245/245候选 `Solve_Succeeded`，numeric failure、infeasible和timeout均为0；求解时间P50为6.84 ms、P95为10.45 ms、最大11.93 ms，低于80 ms门槛。

该结论覆盖Static Sparse和Static Dense三档难度、seed62、每次8秒录制。动态障碍协方差编码也已修改，但仍需在下一阶段通过完整24-run复验，不能用本次静态结果代替动态场景验收。

## 阶段A：根因诊断

原seed61数据包含483条候选，其中112条为 `Maximum_Iterations_Exceeded`。逐候选诊断得到：

1. 候选轨迹消息和提取时的里程计不同步，部分组初始状态偏差最高约0.16 m，动力学rollout缺陷最高约0.33。
2. `radius_m=0.39`本来是0.55 m方箱的外接圆半径，但预测又把长宽写成0.78 m；求解器再取包围盒外接圆后得到0.551 m，相当于重复放大。
3. 旧静态GT协方差使用 `0.01+0.05t`。在2秒处标准差约0.332 m，三倍标准差膨胀约0.995 m；这不符合已知Gazebo静态真值。
4. 失败组所需slack平方中位数为1.60–3.17 m²，超过配置上限1.0 m²。IPOPT经常以迭代上限退出，没有把固定初始状态约束的结构性不可行明确区分出来。
5. 旧teacher使用最后一条逃逸/倒车候选预热，不是稳定的安全停车初值。

诊断证据保存在 `teacher_failure_diagnostics_before.json`，包括每个失败候选的活跃障碍数、初始/轨迹屏障、所需slack、初始偏差、动力学缺陷、求解时间和底层状态。

## 阶段B：实施修复

- 候选控制统一从同步 `ego_state` 重新rollout，消除初始锚点偏差。
- 仿真方箱预测长宽改为 `sqrt(2)*radius`，使求解器恢复正确的0.39 m外接圆。
- 静态GT协方差固定为 `0.0025 m²`；动态障碍使用 `0.0025+0.02t m²`，不再套用旧的过度增长模型。
- 求解前检查固定初始状态约束；所需slack超过上限时直接标记 `Infeasible_Initial_State`，迭代次数为0。
- 普通初值始终由当前ego状态和控制重新rollout。
- 使用最低平均运动状态的受控停车候选预热；仅对 `Maximum_Iterations_Exceeded` 使用该候选有限重试一次。
- 记录每个候选每次尝试的活跃/总障碍数、迭代次数、初值来源和底层返回状态。
- 审计程序增加按场景/难度的outcome、solver status以及求解时间P50/P95/最大值。

## 开发机复验

复验矩阵为Static Sparse、Static Dense × easy/nominal/hard × seed62，单次8秒：

| 场景 | 样本 | 候选 | Success |
|---|---:|---:|---:|
| Static Sparse easy | 7 | 49 | 49 |
| Static Sparse nominal | 5 | 35 | 35 |
| Static Sparse hard | 6 | 42 | 42 |
| Static Dense easy | 5 | 35 | 35 |
| Static Dense nominal | 7 | 49 | 49 |
| Static Dense hard | 5 | 35 | 35 |
| 合计 | 35 | 245 | 245 |

最慢组为Static Dense hard：P50 9.65 ms、P95 11.47 ms、最大11.93 ms。诊断复跑的失败数为0。

## 当前边界与下一步

阶段A/B静态验收已经通过，但尚不批准直接执行72-run正式采集。下一步应使用新预测编码重跑完整8场景×3难度×8–10秒矩阵，特别确认Head-on、Multi Dynamic和Crossing的动态协方差、不可行率与超时率。完整24-run达到每组合至少5帧、numeric failure低于5%、timeout低于1%后，才扩大到多随机种子。

## 证据文件

- `teacher_failure_diagnostics_before.json`：修复前逐候选根因数据。
- `teacher_failure_diagnostics_after.json`：新静态矩阵失败数0。
- `stageb_static_seed62_runs.json`：六个成功bag及样本数量。
- `stageb_static_seed62_raw_audit.json`：35个原始同步样本审计。
- `stageb_static_seed62_teacher_audit.json`：245条teacher标签、checkpoint和求解时间审计。
