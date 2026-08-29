# 阶段 D 完成与阶段 F 启动报告

## 结论

阶段 D 已完成并通过数据门禁。阶段 F 已正式启动，完成冻结数据集、主模型20 epoch训练和独立test run评测；五组消融实验尚未执行，因此阶段 F 状态为“进行中”。

## 阶段 D 完成结果

- 采集矩阵：8场景 × 3难度 × seed64/65/66，共72组，每组30秒。
- bag完整性：72/72通过，均有metadata和非空SQLite数据。
- 同步样本：1929/1929有效，每个场景/难度聚合75–84个样本。
- UniLION特征：1929/1929生成成功，checkpoint SHA-256为`9ba65cfead901ea08db3664108838d4c85bd6621592cc08cca0dfe61af4eb262`。
- teacher候选：共13503条，其中13202 success、215 infeasible、78 numeric failure、8 timeout。
- numeric failure率0.58%，timeout率0.06%，分别低于5%和1%门槛。
- 求解时间P50 3.15 ms、P95 8.27 ms、最大87.84 ms；超过80 ms的8条被严格标为timeout。

## 冻结数据集

- 数据版本：`r680_staged_v1`。
- 冻结manifest SHA-256：`c9cef0ebc1f6fc5d6499542e8293ed9166135077c654a73a534b6b7605bd4d7d`。
- 按完整seed隔离：seed64=train、seed65=validation、seed66=test。
- train/validation/test各643样本，不存在跨seed帧泄漏。
- 数据版本文件同时记录源manifest哈希、划分配置哈希、UniLION checkpoint哈希和Git提交号。

## 阶段 F 主训练

- 联合训练对象：候选轨迹头和安全评估头。
- 配置：20 epoch、batch size 8、AdamW、学习率`3e-4`、随机种子680。
- 最佳checkpoint：epoch 18，开发机路径`.tools/training/r680_staged_v1_main/best.pt`。
- checkpoint格式：v2，包含模型状态、优化器状态、训练配置及SHA-256、manifest SHA-256、数据版本SHA-256、Git提交号和验证指标。
- 训练配置SHA-256：`26327452140403a3b3db25b9f482e28e37dddf850ecf232218078c2f9c5785a3`。

## 独立 test 评测

测试集为未参与训练或模型选择的seed66，共643个样本：

- 总loss：0.70496。
- 控制MAE：0.11934。
- 候选排序准确率：71.85%。
- 可行性准确率：98.76%。

## 未完成内容

阶段 F 尚需执行并统一汇总以下消融：

1. 无UniLION特征。
2. 无D-CBF teacher/闭环约束。
3. 无障碍协方差teacher标签。
4. 不同候选数量。
5. 不同障碍筛选数量。

其中后四项会改变teacher或候选数据，必须生成版本化派生数据集，不能只在网络输入端切换开关。
