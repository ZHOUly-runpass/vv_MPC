# UniLION 阶段1执行报告

日期：2026-08-25  
开发机：`guanpc`，RTX 4090 24 GB  
原始证据：`reports/unilion_stage1_forward.json`

## 结论

阶段1的工程目标已经完成：官方配置能够构建冻结的单帧LiDAR子网，911项参数/缓冲区严格匹配，合成C16格式点云已在开发机完成真实CUDA前向、BEV特征hook、有限值、方差、显存、时延和重复性验证。

尚有一个外部数据验收项：开发机没有 nuScenes `samples/LIDAR_TOP` 原始文件，无法把合成输入描述为官方真实样例。执行脚本已支持 `--nuscenes-sample`，取得合法数据后可直接补跑。该缺口不影响确认代码和CUDA链可执行，但真实C16冻结门继续关闭。

## 实现内容

- `UniLionPointFieldContract`：严格映射 `x,y,z,intensity,ring`，校验字段数量、唯一性、有限值和C16 ring范围。
- `UniLionFrozenBackbone`：从官方配置独立构建 voxel encoder、Lion3D backbone、HeightCompression、BEV backbone 和 neck。
- 严格权重加载：对参数和缓冲区名称集合、形状逐项校验，不接受 missing/unexpected key；兼容检查函数只允许明确的旧版spconv卷积核布局转换。
- 冻结检查：所有模块使用 eval 模式，所有参数 `requires_grad=False`。
- BEV hook：记录 sparse backbone、height compression、BEV backbone 和 neck 的形状、有限值、均值和方差。
- 前向脚本：支持确定性合成C16格式输入及外部 nuScenes 五浮点样例，输出JSON证据。

## 权重

- 完整LCT：`14377fe6656ed487f40ad6af3161055bcc68956394599845b1c6f234a1b41256`，400672958字节。
- 阶段1LiDAR子集：`9ba65cfead901ea08db3664108838d4c85bd6621592cc08cca0dfe61af4eb262`，62833816字节。
- 子集共911项：voxel encoder 12、LiDAR backbone 767、BEV backbone 114、neck 18。
- 权重来源仍是多模态LCT，不是C16专用模型，只能作为初始化候选。

## 开发机结果

- 输入：12000点，16个ring，确定性合成C16字段格式。
- 有效体素：7792。
- 最终BEV：`[1,384,180,180]`。
- 特征有限值：通过。
- 最终特征方差：约 `0.00226196`。
- 模型前向中位数：约56.64 ms。
- 包含GPU到CPU复制的总耗时中位数：约70.75 ms。
- 峰值CUDA已分配内存：491505152字节，约468.7 MiB。
- 重复前向最大绝对差：约 `3.72e-4`，通过 `1e-3` 阈值；spconv/Mamba稀疏CUDA路径不是逐比特确定性。

四级特征形状：

- Sparse LiDAR backbone：`[30156,128]`。
- HeightCompression：`[1,256,360,360]`。
- BEV backbone：`[1,128,360,360]`、`[1,128,180,180]`、`[1,256,90,90]`。
- BEV neck：`[1,384,180,180]`。

## 字段结论

官方配置 `load_dim=5/use_dim=[0,1,2,3,4]` 对应 `x,y,z,intensity,ring`。C16的相对 `time` 字段保留给上游去畸变，但不会填入官方第五维。Voxel encoder首层11维的来源是：5维原始点、3维点到pillar均值偏移、3维点到体素中心偏移，不要求PointCloud2原生提供11个字段。

## 未解除的验证门

- `pure_lidar_checkpoint_verified=false`：权重来自LCT多模态训练。
- `c16_forward_100_frames_verified=false`：当前仅完成单帧合成格式验证。
- `frozen_backbone_verified_on_c16=false`：没有真实C16数据证据。
- 官方nuScenes真实样例前向：待合法取得数据后执行。

因此阶段1结果可以进入仿真16线100帧阶段，但不能直接解锁实车控制。
