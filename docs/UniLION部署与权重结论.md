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

官方要求 Python 3.9、PyTorch 2.5.0+cu124、CUDA 12.4，并编译其修改版 MMCV、MMDetection3D、Mamba 和项目 CUDA 扩展。开发机当前为 Python 3.10、系统 nvcc 11.5，不能在现有 ROS 环境内直接编译。

`environments/unilion_environment.yml` 定义了隔离环境基础。后续需安装 Miniforge/Conda，再依次编译：

1. `third_party/UniLION/mmcv`
2. `third_party/UniLION/mmdetection3d`
3. `third_party/UniLION/projects/mmdet3d_plugin/models/ops/mamba`
4. `third_party/UniLION/projects`

官方 `requirement.txt` 同时包含 CUDA 11.6 的 `spconv/cumm` 与 CUDA 12.4 说明，不应整文件盲装；必须先做最小前向依赖闭包。
