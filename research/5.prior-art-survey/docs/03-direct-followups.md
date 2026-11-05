# 同组接续工作（直接相关）

- 创建：2026-11-05 · claude code (local)
- 判定标准：作者与 MobileIE 团队重合（Hailong Yan、Ao Li、Xiangtao Zhang、Le Zhang、Ce Zhu 等），或明确以 MobileIE 为基座/基线。
- 关键结论：**5 篇同组接续 + 3 个竞赛报告，全部是图像任务；没有一篇做视频/时序一致性。**

## 1. 五篇同组接续

### 1.1 Revisiting Lightweight Low-Light Image Enhancement: From a YUV Color Space Perspective

| 项 | 内容 |
| --- | --- |
| 作者 | Hai-Long Yan, Shi-Ce Liu, Xiang-Tao Zhang, Lujian Yao, Feng Yang, Jin-Wei Chen, Bo Li |
| 出处 | arXiv 2601.17349（2026） |
| 任务 | 单图轻量低光增强（L3IE） |
| 方法 | 频域分析 → YUV 空间；Y 通道 Dual-Stream Global-Local Attention，UV 通道 Y-guided Local-Aware Frequency Attention，Guided Interaction 融合 |
| 结果 | 多基准 SOTA，参数量显著更低 |
| 与 MobileIE 关系 | 同第一作者，最可能的“往后做了一点” |
| 重复风险 | **中高**：若我们宣称“轻量低光增强的新结构”，会被拿来对比；但它不做视频/时序 |
| 应对 | 引用并明确区分：我们不改单图结构，解决帧间一致性 |

### 1.2 Towards Lightest Low-Light Image Enhancement Architecture for Mobile Devices（LiteIE）

| 项 | 内容 |
| --- | --- |
| 作者 | Guang-Rui Bai, Hai-Long Yan, Wen-Hai Liu, Yahui Deng, Er-Bao Dong |
| 出处 | Expert Systems with Applications（2025）；arXiv 2507.04277 |
| 任务 | 单图无监督低光增强 |
| 方法 | 两层卷积特征提取 + 无参数迭代复原模块 + 无监督损失（曝光/边缘/多尺度颜色一致性） |
| 结果 | LOL 19.04 dB；**58 参数**；骁龙 8 Gen 3 上 4K 30 FPS |
| 与 MobileIE 关系 | 同作者；效率指标上与 MobileIE 直接竞争 |
| 重复风险 | **高（效率维度）**：如果只说“更轻”，会被 LiteIE 压住；必须强调时序一致性 |
| 应对 | 作为效率强基线引用；说明 58 参数是图像模型，不处理帧间闪烁 |

### 1.3 IGDNet: Zero-Shot Robust Underexposed Image Enhancement via Illumination-Guided and Denoising

| 项 | 内容 |
| --- | --- |
| 作者 | Hailong Yan, Jun-Jian Huang, Ting-Wen Huang |
| 出处 | IEEE Transactions on Artificial Intelligence（2025）；arXiv 2507.02445 |
| 任务 | 零样本单图欠曝增强 |
| 方法 | 光照/反射分解 + 光照引导的像素自适应校正 + 噪声对迭代 |
| 结果 | 四个公开数据集，PSNR 20.41 dB，优于 14 个无监督方法 |
| 与 MobileIE 关系 | 同第一作者；单图、无监督路线 |
| 重复风险 | 低～中：问题不同，但会被引用来质疑“为什么不用零样本” |
| 应对 | 明确我们是有监督 LLE + 时序，训练数据/目标是两件事 |

### 1.4 Real-Time Underwater Image Enhancement via Frequency-Guided Dual-Path Attention（FGDPA）

| 项 | 内容 |
| --- | --- |
| 作者 | Le Zhang, Ao Li, Ce Zhu |
| 出处 | arXiv 2606.30314（2026） |
| 任务 | 实时水下图像增强 |
| 方法 | MBRConv-DCT（固定 DCT 先验的多分支重参数化卷积）+ Frequency-Guided Dual-Path Attention；**结构重参数化后零额外推理开销** |
| 结果 | 4.23K 参数，600+ FPS，SOTA |
| 与 MobileIE 关系 | 同组；把 MBRConv 思路迁移到 UIE |
| 重复风险 | **中高（方法学）**：如果我们的时序模块也走“重参数化、零推理开销”，会被认为与 FGDPA 同源 |
| 应对 | 区分：FGDPA 是空间/频域结构，解决图像质量；我们解决帧间一致性，且可复用其重参数化经验 |

### 1.5 DRNet: All-in-One Image Restoration via Prior-Guided Dynamic Reparameterization

| 项 | 内容 |
| --- | --- |
| 作者 | Ao Li, Xiao-Ning Liu, Sheng Li, Ya-Peng Du, Zhen Long, Lei Luo, Le Zhang, Ce Zhu |
| 出处 | IEEE TMM（2026）；arXiv 2605.08627 |
| 任务 | all-in-one 图像复原 |
| 方法 | 初始化阶段动态重配置 + Task-Specific Modulator + 小波编码器；避免逐输入动态估计开销 |
| 结果 | 五个复原任务 SOTA，参数效率高 |
| 与 MobileIE 关系 | 同组；重参数化路线的延伸 |
| 重复风险 | 中：如果我们的卖点是“动态重参数化”，会被认为重复 |
| 应对 | 强调时序维度；重参数化只作为实现手段，不作为主要 novelty |

## 2. 三个竞赛/基线报告

| 工作 | 关系 | 对我们的意义 |
| --- | --- | --- |
| *Learned Smartphone ISP on Mobile GPUs: Mobile AI 2026 Challenge Report*（arXiv 2611.05008） | 团队成员参与，MobileIE 作为基线 | 说明端侧效率评估已是公开赛道；需报告延迟/能耗 |
| *Efficient Image Denoising on Smartphone GPUs: Mobile AI 2026 Challenge Report*（arXiv 2611.00974） | MobileIE 作为基线 | 同上 |
| *NTIRE 2026 Challenge on Efficient Low Light Image Enhancement*（arXiv 2605.02212） | MobileIE 作为基线 | 高效 LLIE 竞争激烈；投稿需对齐其指标口径 |

## 3. 对“重复性”的判断

| 维度 | 是否已被同组做过 | 结论 |
| --- | --- | --- |
| MobileIE 单图结构改进 | 是（YUV、FGDPA、DRNet） | 不要作为主卖点 |
| 更轻量的单图增强 | 是（LiteIE 58 参数） | 不要只比参数量 |
| 重参数化零推理开销 | 是（FGDPA、DRNet、MobileIE 本身） | 可作为实现手段，不作为 novelty |
| **视频帧间一致性/闪烁** | **否** | **仍是空白** |
| **极轻量单图模型的训练期时序约束** | **否** | **主要候选卖点** |

## 4. 建议引用方式

- Introduction 中把 YUV、LiteIE、FGDPA、DRNet 列为“同组在单图/效率方向的后续工作”，一句话说明它们**均针对单帧**；
- 方法部分引用 FGDPA/DRNet 的重参数化经验，说明我们的时序分支（若需要）如何保持推理零开销；
- 实验部分至少把 LiteIE 作为效率参照（58 参数、4K 30 FPS），把 FGDPA 作为 4.23K 参数的 UIE 参照；
- 不要声称“首次轻量低光增强”或“首次重参数化增强”，这些已被同组占位。

## 5. 待核实

- [ ] YUV 论文是否包含视频实验/时序损失（摘要未见，需读全文）；
- [ ] DRNet 是否可用于视频帧序列（方法上可能，但论文未涉及）；
- [ ] LiteIE 的无监督一致性损失是否会被审稿人误认为“时序一致性”（需在读全文后准备区分话术）。
