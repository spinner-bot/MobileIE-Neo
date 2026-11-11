# R35补充：公开权重训练来源与开发片重叠

记录：2026-11-12T05:25:24+08:00；作者：Codex  (sqs_A100)。触发：同底座迁移评价运行期间补齐公开预训练来源交叉核验，不修改正在运行的原协议/代码，不重启或筛选当前结果。

固定[官方README](https://github.com/m-tassano/fastdvdnet/blob/c8fdf6182a0340e89dd18f5df25b47337cbede6f/README.md)说明使用DAVIS2017训练，并链接[作者训练清单](https://gist.github.com/m-tassano/27c1ef00ca42a8e50c2cee8a4205e559)。取得清单固定revision b7f358c73d0f560ed1b2f0f33b5cc7fd8c25c401，与已保存官方split作ID交集：[结果](../data/external-davis/fastdvdnet-training-overlap.json)为90训练视频，覆盖TRAIN60、VAL30及R30三开发片bear/lucia/walking。

该清单不是精确clipped-noise权重的逐帧训练日志，不能证明每张图片都实际被抽样；但足以要求将三开发片视为公开预训练来源重叠，不能当未见视频泛化或公平重训SOTA结论。此前R30已明确DEV而非最终测试，本次补正其针对公开权重的更强限制。window/five_current仍可回答同权重真实时间信息问题，两个调用有同一预训练来源；R35按原固定范围继续完整运行。

R34训练仍严格排除三开发片，原独立空间/因果控制不因本发现改变。但未来若与官方FastDVDnet权重一起评价，DAVIS VAL也不能当所有方法都未见的最终视频；应另外保留与上述90训练ID完全不重叠的公开来源，或为所有方法在相同新划分重训。先只核查来源元数据，不提前读最终像素。训练池与基线来源的二级交叉排除应写进最终基准协议。

汇总器在包含published阶段时强制读取此限制并显示在表中。R35原启动config/协议摘要作为历史事实保留，补充限制经本文件与结果表传播；不回写已经启动时的来源记录。
