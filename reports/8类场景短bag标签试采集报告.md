# 8类场景短 bag 标签试采集报告

日期：2026-08-28  
矩阵：8 场景 × easy/nominal/hard × seed 61 × DWB  
单次录制：4 s；提取：1 Hz；时间网格：2.0 s / 0.1 s / 21 状态点

## 验收结论

短 bag 采集、同步帧提取、schema 1.0、UniLION 特征和 teacher 文件完整性全部通过；teacher 标签质量暂不通过，不能据此直接扩大到正式多种子采集。

| 环节 | 结果 | 状态 |
|---|---:|---|
| 成功场景组合 | 24/24 | 通过 |
| 有效同步原始帧 | 69/69 | 通过 |
| UniLION 特征 | 69/69，`[384,32,32]` | 通过 |
| checkpoint 哈希一致 | 69/69 | 通过 |
| teacher 候选 | 483 | 完成 |
| teacher success | 364（75.36%） | 不足 |
| infeasible | 7（1.45%） | 保留为有效失败标签 |
| numeric failure | 112（23.19%） | 阻塞正式扩大采集 |
| timeout | 0 | 通过 |

使用的 UniLION LiDAR checkpoint SHA-256 为 `9ba65cfead901ea08db3664108838d4c85bd6621592cc08cca0dfe61af4eb262`，特征配置 SHA-256 为 `5ebea3991a61298dd9df62911c4dec50f2c5ec4b2e0b981f1746c2d445ab6dd9`。

## 场景审计

- Empty、Crossing、局部极小陷阱三档均为全 success。
- Head-on nominal/hard 全 success；easy 有 7 条 `Maximum_Iterations_Exceeded`。
- Multi Dynamic nominal/hard 全 success；easy 有 7 条 `Maximum_Iterations_Exceeded`。
- Narrow Passage nominal/hard 全 success；easy 有 7 条明确的 `Infeasible_Problem_Detected`，属于可保留的不可行标签。
- Static Sparse easy 全 success；nominal 为 14 success + 7 数值失败；hard 28 条全部数值失败。
- Static Dense easy/nominal 全部为数值失败；hard 为 7 success + 14 数值失败。

112 条数值失败的底层状态均为 `Maximum_Iterations_Exceeded`，不是 schema 损坏、非有限数组或 checkpoint 不一致。这表明当前主要问题在多静态障碍条件下的 CasADi/IPOPT 收敛，而不是 rosbag 或 UniLION 链路。

## 难度真实性

原始样本中静态稠密场景的障碍预测数量随难度出现 6、9、12；静态稀疏场景出现 1、2、3。狭窄通道和陷阱长墙只编码在 costmap，没有错误进入圆形动态障碍约束。全部候选形状为 `[7,21,5]`、控制为 `[7,20,2]`、障碍预测时间长度为21。

## 运行问题及修复

开发机上其他 ROS 2 任务与采集任务最初共同使用 Domain 0，残留 `/controller_manager` 发现信息导致5次 preflight 失败；失败 bag 未进入成功数据集。录制入口现按场景、难度、seed、控制器生成确定性的独立 `ROS_DOMAIN_ID`，并启用 `ROS_LOCALHOST_ONLY=1`。同时为 ROS launch 清理增加 INT、TERM、KILL 分级时限，并为批量脚本增加成功组合续跑，重试后24个组合全部成功。

## 正式采集前必须完成

1. 针对静态稀疏 hard 和静态稠密三档分析初值、可达障碍筛选、D-CBF 松弛和 IPOPT 最大迭代收敛；数值失败必须显著下降，不能把它们改名为不可行。
2. 将短录制时长由4 s提高到至少8–10 s。当前 Multi Dynamic nominal/hard 各仅提取1帧，足以验证链路，不足以估计标签分布。
3. 修复后重新执行本24-run矩阵，并设定扩大采集门槛，例如每个组合至少5个有效样本、numeric failure低于预先确定阈值。
4. 通过门槛后再扩展到3个或更多随机种子；当前 seed61 数据保持 `trial` split，不并入最终 train/val/test。

## 证据

- `collection_trial_seed61_runs.json`：24个最终成功 bag 和每个 bag 样本数。
- `collection_trial_seed61_raw_audit.json`：69个原始样本的结构、障碍数量和元数据分组。
- `collection_trial_seed61_teacher_audit.json`：checkpoint、样本形状、场景分组 outcome 和底层求解状态。
