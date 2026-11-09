# 1. 前期准备（Preparation）

## 目标

明确研究方向并完成前期调研与复现准备：以 MobileIE 为基础，解决该轻量图像增强模型应用于视频时的帧间闪烁问题。

## 背景

- **基座工作**：*MobileIE: An Extremely Lightweight and Effective ConvNet for Real-Time Image Enhancement on Mobile Devices*（ICCV），极轻量 CNN，面向手机端实时图像增强（去噪、增亮等），有开源代码与手机模拟器，单卡 4090 可训练。
- **问题**：模型按单张图像设计，逐帧处理视频时帧间差异过大，输出视频出现闪烁（本应是连续的帧间映射）。
- **性质**：0→1 型问题——已有问题、尚无解法，解决即可成文，非刷指标型竞赛。
- **判断**：需从训练阶段入手，配合模型结构/系统设计，后期微调无法根治。

## 相关材料

工作文档见 [`docs/`](docs/)。

| 文件 | 说明 |
| --- | --- |
| `docs/Standard recording 67 - Transcription.txt` | 与导师讨论的会议转写（方向、实验经验、投稿常识） |
| `docs/wechat-message.txt` | 基座论文标题 |

## 待办

- [ ] 精读 MobileIE 论文，跑通官方代码与单图推理
- [ ] 复现并录制视频闪烁现象（作为 baseline demo）
- [ ] 调研视频时序一致性方法（temporal loss、光流/特征对齐、recurrent 结构等）在轻量网络上的可迁移性
- [ ] 输出一份问题定义 + 复现方案 + 文献清单（置于本目录）

## 备注

- 实验建议：在跨领域综合数据集上验证，写作时选定具体领域叙事，不强行宣称通用性。
- 周期预期 3–6 个月，按周推进产出进展。
