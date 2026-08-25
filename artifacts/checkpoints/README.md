# UniLION checkpoint

权重文件不提交 Git。官方下载链接：

`https://drive.google.com/file/d/18fpw-EJ-eJikVPczoRqyhLnXjJzjOnpv/view`

预期文件名：`unilion_swin_384_seq_e2e.pth`

该权重对应官方 LCT（LiDAR + Camera + Temporal）端到端配置，不是纯 LiDAR/C16 专用权重。

- SHA-256：`14377fe6656ed487f40ad6af3161055bcc68956394599845b1c6f234a1b41256`
- 大小：`400672958` 字节

抽取的确定性 LiDAR 初始化子集：`unilion_lidar_backbone_init.safetensors`

- 包含 797 个参数/缓冲区：voxel encoder 12、backbone 767、neck 18。
- SHA-256：`89ab91f6d33ae209331a8dbc4f1a58fe392beca790f21086e87a2bf9195b3a29`
- 大小：`42975912` 字节
- 来源与统计保存在同名 `.json` sidecar；SafeTensors 本体不写 metadata，以保证跨机器确定性。
- 仅作为初始化候选，未通过 C16 验证。
