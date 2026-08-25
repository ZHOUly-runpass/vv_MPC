# UniLION checkpoint

权重文件不提交 Git。官方下载链接：

`https://drive.google.com/file/d/18fpw-EJ-eJikVPczoRqyhLnXjJzjOnpv/view`

预期文件名：`unilion_swin_384_seq_e2e.pth`

该权重对应官方 LCT（LiDAR + Camera + Temporal）端到端配置，不是纯 LiDAR/C16 专用权重。

- SHA-256：`14377fe6656ed487f40ad6af3161055bcc68956394599845b1c6f234a1b41256`
- 大小：`400672958` 字节

抽取的 LiDAR 初始化子集：`unilion_lidar_backbone_init.pth`

- 包含 797 个参数/缓冲区：voxel encoder 12、backbone 767、neck 18。
- SHA-256：`83985840ab03b2543ad06507941d5171ae447609119e8b2e44201afb8f9ca7d9`
- 大小：`43157708` 字节
- 仅作为初始化候选，未通过 C16 验证。
