# UniLION 部署与权重结论

## 官方来源

- 代码：`https://github.com/happinesslz/UniLION`
- 固定提交：`b75c28415ed53d5474bcb9be776d032c66ec432b`
- 官方 LCT 权重：Google Drive 文件 `18fpw-EJ-eJikVPczoRqyhLnXjJzjOnpv`
- 完整权重 SHA-256：`14377fe6656ed487f40ad6af3161055bcc68956394599845b1c6f234a1b41256`
- 完整权重大小：`400672958` 字节

官方论文和 README 描述了 L、LT、LC、LCT 四类配置，但结果表只给 LCT 模型下载链接，L/LT 没有公开 checkpoint。LCT 权重包含 `pts_voxel_encoder`、`pts_backbone` 等 LiDAR 参数，可以抽取作为初始化候选，但不是 C16 专用或已验证的纯 LiDAR checkpoint。

## 许可证风险

固定提交的仓库根目录没有 `LICENSE`。在作者明确授权前，本项目只以 Git submodule 指向官方仓库，不复制发布源码；权重不进入 Git，只保存官方 URL、哈希和结构元数据。

## C16 适配差异

官方配置使用 nuScenes 范围 `[-54,-54,-3,54,54,5]`、体素 `[0.3,0.3,0.25]`，点输入 `load_dim=5/use_dim=[0..4]`。Voxel encoder 的首层权重为 `[64,11]`，11 是 pillar 内构造后的特征维数，不等于 C16 PointCloud2 必须原生提供 11 个字段。

C16 适配必须验证：

1. 将 `x,y,z,intensity,ring,time` 映射为官方点输入，缺少语义字段不得填伪值。
2. 局部导航范围裁剪到约 `x∈[-2,8], y∈[-5,5]`，重新确定体素大小。
3. 用 C16 点云跑至少 100 帧，检查有限值、空体素率、特征方差和延迟。
4. 冻结 voxel encoder/backbone，训练本项目 feature adapter；必要时只解冻末端块。
5. 抽取权重只能标为 `initialization_candidate`，不能关闭 `frozen_backbone_verified_on_c16`。

## 开发机环境

官方要求 Python 3.9、PyTorch 2.5.0+cu124、CUDA 12.4，并编译其修改版 MMCV、MMDetection3D、Mamba 和项目 CUDA 扩展。开发机宿主仍为 Python 3.10、系统 nvcc 11.5，因此没有污染 ROS 环境，而是在项目 `.tools/envs/unilion` 建立了隔离环境。

截至 2026-08-25，开发机 RTX 4090 上已经完成：Python 3.9、PyTorch 2.5.0+cu124、环境内 nvcc 12.4.131、MMCV 1.7.0 CUDA ops、MMDetection3D 1.0.0rc6、Mamba 1.1.1、`causal-conv1d` 1.5.0.post8 和 UniLION 项目扩展。`scripts/smoke_unilion_dev.py` 的 CUDA/导入检查全部通过。

`environments/unilion_environment.yml` 与 `scripts/setup_unilion_dev.sh` 固化了环境和以下编译顺序：

1. `third_party/UniLION/mmcv`
2. `third_party/UniLION/mmdetection3d`
3. `third_party/UniLION/projects/mmdet3d_plugin/models/ops/mamba`
4. `third_party/UniLION/projects`

官方 `requirement.txt` 同时包含 CUDA 11.6 的 `spconv/cumm` 与 CUDA 12.4 说明，因此未整文件盲装；当前使用 `spconv-cu120 2.3.6` 与 Torch/CUDA ABI 对应的 `torch-scatter` wheel。Conda 的 `cuda-version/cuda-cccl` 已固定为 12.4，避免求解出 13.x 头文件导致 nvcc 12.4 编译失败。

当前通过的是环境、CUDA 扩展和完整插件导入，不等于 C16 推理验收。仍需官方 nuScenes 样例前向、LiDAR 子网严格加载、C16 点字段适配及 100 帧验证。
