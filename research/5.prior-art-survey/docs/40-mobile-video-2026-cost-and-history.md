# 2026移动视频工作：预算约束不能简化成必须单帧

定向核查：2026-11-12T18:57:34.853148+08:00；作者：Codex  (sqs_A100)。两篇公开作者PDF存zrh自有data，获取与实际阅读层级分别记录；未复现网络/端侧速度，未取得新视频像素。

## CVPR2026 raw-to-raw

[CVF正式条目](https://openaccess.thecvf.com/content/CVPR2026/html/Pochimireddy_Efficient_Real-Time_Raw-to-Raw_Denoising_for_Extreme_Low-Light_Ultra_HD_Video_CVPR_2026_paper.html)核对标题/发表。实际读[作者全文](https://lumen-cvpr26.github.io/raw-to-raw-extreme-low-light-enhancement/assets/pdfs/paper.pdf)§3.1–3.4、§4.1/4.4与表4/5：RAW预去马赛克，S2D/mRLFB；二帧扩首卷积，重参数化未折叠掉前帧依赖。真实监督含90帧平均后模型去残噪，运动部分为合成，非全部真实自由动态GT。手机Hexagon NPU表5报告Model BRQ 12.91ms/244mA，FP16/INT16须区分；非实时比较者电流由时长线性外推，不都是相同实测。代码/数据在作者页仍coming soon；RAW值/GT/端侧链不同，不移入本项目PSNR或CPU排名。

## UbiEnlight，IMWUT2026

[大学正式记录](https://scholars.cityu.edu.hk/en/publications/ubienlight-user-perception-aware-real-time-low-light-video-enhanc/)核对DOI10.1145/3832024。实际读[作者PDF](https://zhouzimu.github.io/paper/ubicomp26-wang.pdf)§4.1、§5.1/5.3.3及表4–6：扩散/时序注意力加关键帧latent缓存，推理保留历史。§5.1输入20FPS、270×480；§5.3.3明确输入帧率和增强吞吐不同。表5 Pixel7 CPU76.47–159.97ms，表6 Jetson GPU66.16–75.51ms，不等于每帧完整20/30FPS或1080p/4K承诺。眼镜由配对手机计算，不能作眼镜本体算力。未核验其代码/权重可复现入口，也不采用该文对StableLLVE推理依赖的归类替代已读官方源码。

## 对本项目的判断（非上述论文结论）

“手机实时因此只能训练时用时间、推理完全单帧”不成立。应分别约束目标分辨率/每帧计算/状态内存/精度/等待帧数及实测吞吐、时延、功率；当前4075参数不能代替这些指标。推理零新增成本可保留为一个子目标，不能据它排除全部轻因果骨干。与此同时，双帧输入、S2D、重参数化、缓存或频域条件本身都已有先例。

不据此重启R8双帧/R34低分辨率/R37低秩适配或已停止的R48等网格，不凭新论文名拼模块。R47任务适配与R50固定BN控制收齐之后，应将空间质量可信度、真实历史独立作用与明确端侧预算共同用于收敛下一机制；有效创新、外部自然夜间、手机实测和独立最终测试仍未完成。
