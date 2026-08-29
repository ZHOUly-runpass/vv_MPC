# 阶段 C/D/E 执行报告

## 结论

- 阶段 C 已通过，可以进入多随机种子正式采集。
- 阶段 D 已真实启动并完成首批 12/72 个 30 秒 bag，剩余 60 组尚未执行，不能标记为完成。
- 阶段 E 的四控制器统一接口冒烟通过；DWB、MPPI 和 Vanilla D-CBF 可继续扩大评测，Proposed 因学习 checkpoint 未激活，正式公平基线仍被阻塞。

## 阶段 C：完整 24-run 标签门禁

- 矩阵：8 类场景 × easy/nominal/hard × seed63，单次 10 秒。
- bag：24/24 成功；曾有两次 Nav2 costmap 瞬时未发布，重试后通过且未纳入成功证据。
- 数据：187/187 个 schema 1.0 样本有效，每组 6–10 个，满足每组至少 5 个的门槛。
- UniLION：187/187 特征成功，checkpoint SHA-256 为 `9ba65cfead901ea08db3664108838d4c85bd6621592cc08cca0dfe61af4eb262`。
- teacher：1309 条候选中 1274 success、35 infeasible、0 numeric failure、0 timeout。
- 时延：P50 3.05 ms、P95 8.61 ms、最大 16.10 ms，均低于 80 ms deadline。
- 35 条不可行集中在 `head_on/easy`、`multi_dynamic/easy`、`multi_dynamic/nominal` 和 `narrow_passage/easy`；它们由求解器明确报告为 `Infeasible_Problem_Detected`，未混同数值失败。

## 阶段 D：多种子正式采集进度

- 正式矩阵：8 类场景 × 3 难度 × seed64/65/66，共 72 组，单次 30 秒。
- 当前完成 12/72：Empty 的三难度×三种子，以及 Static Sparse/easy×三种子。
- 当前成功 bag 均包含 `metadata.yaml` 和非空 SQLite 数据文件；剩余 60 组未采集。
- 续跑命令使用 `.tools/collection_staged_dwb/runs.jsonl` 与 `--resume-successes`，不会重复已成功组合。

## 阶段 E：四控制器统一短闭环

场景为 Static Sparse/nominal、seed67、每组 8 秒。四组均有完整 topic、控制器身份、非零控制、里程计位移和 benchmark 状态，且无碰撞：

| 控制器 | 里程计位移 | 最小间距 | 碰撞 | 当前认定 |
|---|---:|---:|---|---|
| DWB | 2.01 m | 1.37 m | 否 | 接口冒烟通过 |
| MPPI | 1.88 m | 1.01 m | 否 | 接口冒烟通过 |
| Vanilla D-CBF | 1.09 m | 1.77 m | 否 | 确定性安全控制器冒烟通过 |
| Proposed | 0.37 m | 2.91 m | 否 | 回退接口通过，正式基线 blocked |

这组 8 秒测试只验证闭环与记录接口，不用于比较到达率或算法优劣；四组均未在短窗口内到达终点。

## 执行中修复

1. rosbag 改为显式启动、定时、进程组 SIGINT 收尾并校验 `metadata.yaml`/`ros2 bag info`，避免将强杀返回码误判为有效数据。
2. 批处理被中断时会清理当前子进程组，避免遗留 Gazebo 实例污染下一组。
3. collection preflight 支持有限重试，但不降低 topic 频率、TF、场景状态或磁盘门槛。
4. 启动等待覆盖全部必需 topic，消除 DDS 旧发现缓存造成的提前通过。

## 下一步

1. 从 12/72 断点续跑阶段 D 剩余 60 组。
2. 完成后按 run 划分 train/validation/test，再批量生成 UniLION 特征和 teacher 标签。
3. 为 Proposed 接入真实训练 checkpoint 与推理桥；在 `learned_checkpoint_active=true` 且哈希可审计前，不进行正式四基线结论性比较。
4. 四控制器使用相同 8×3×多种子矩阵，统一统计成功率、碰撞率、最小间距、到达时间、平滑度和端到端延迟。
