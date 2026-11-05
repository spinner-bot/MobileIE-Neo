# MobileIE 引用全表（截至 2026-11-05）

- 创建：2026-11-05 · claude code (local)
- 数据：`data/citations_s2.json`（33 条）、`data/citations.csv`
- 相关性为人工标注：高 = 可能构成直接重复/强对比；中 = 相邻方向；低 = 仅工程引用或无关。

## 1. 全表

| # | 年份 | 标题 | 类别 | 相关性 | 说明 |
| --- | --- | --- | --- | --- | --- |
| 1 | 2026 | Learned Smartphone ISP on Mobile GPUs: Mobile AI 2026 Challenge Report | 竞赛 | 中 | MobileIE 作为端侧 ISP 基线 |
| 2 | 2026 | Efficient Image Denoising on Smartphone GPUs: Mobile AI 2026 Challenge Report | 竞赛 | 中 | MobileIE 作为端侧去噪基线 |
| 3 | 2026 | MorphSIG: Subject-Driven Image Generation via Decoupled Anchoring and Feature Transport | 图像生成 | 低 | 同作者组；pseudo-video 是生成技巧，与闪烁无关 |
| 4 | 2026 | CAS-HDPA-Yolov11: a dual attention enhanced network for monocular human height estimation | 检测 | 低 | 借用 HDPA 注意力命名 |
| 5 | 2026 | Co-umamba: Collaborative global–local modeling for precise apple leaf disease segmentation | 分割 | 低 | 无关 |
| 6 | 2026 | Beyond Pipeline Mimicry: Expert-level Aesthetic ISP via Reward-Guided Flow | ISP | 中 | 图像 ISP，非视频 |
| 7 | 2026 | Zero-Shot Illumination and Noise Estimation for Edge-Deployable Low-Light Image Enhancement in IoT Systems | 低光增强 | 中 | 端侧零样本 LLIE |
| 8 | 2026 | HDS-Net: A Hybrid Deformable-Swin Network for Dual-Domain Sparse-View CT Reconstruction | 医学 | 低 | 无关 |
| 9 | 2026 | A Multi-Modal Fusion Network of Visible–Ultraviolet Features for Detecting Aero-Engine Blade Microcracks | 检测 | 低 | 无关 |
| 10 | 2026 | AI-Driven Illumination-Adaptive Enhancement for Robust Mobile AR Games | 应用 | 中 | 在 AR/vSLAM 视频管线中逐帧调用 MobileIE + 自适应触发；未处理时序一致性 |
| 11 | 2026 | EPIDIR: Integrating external frequency prompts and internal statistical decoupling for infrared image restoration | 红外 | 低 | 无关 |
| 12 | 2026 | Energy-Aware Deployment of Quantized Shallow-Water Image Enhancement on an Edge NPU for Underwater IoT Imaging | 部署 | 中 | 强调延迟/能耗而非 params/FLOPs，方法学可借鉴 |
| 13 | 2026 | Real-Time Underwater Image Enhancement via Frequency-Guided Dual-Path Attention | 同组接续 | **高** | Le Zhang / Ao Li / Ce Zhu；图像任务 |
| 14 | 2026 | AnyMod-LLVE: Low-Light Video Enhancement with Modality-Agnostic Inference | 视频 | **高** | 唯一明确的视频 LLVE 引用；多模态、重模型 |
| 15 | 2026 | Frequency-Domain Optimized Laplacian Pyramid Network for Weld DR Image Enhancement | 增强 | 低 | 无关 |
| 16 | 2026 | CS³-Diff: Collaborative spatio-spectral-scale guided diffusion for low-light image enhancement | 低光增强 | 中 | 扩散模型，重 |
| 17 | 2026 | MAEL-DET: a multi-scale adaptive enhanced lightweight network for a welding defect detection algorithm | 检测 | 低 | 无关 |
| 18 | 2026 | DRNet: All-in-One Image Restoration via Prior-Guided Dynamic Reparameterization | 同组接续 | **高** | Ao Li 等；动态重参数化，图像 all-in-one |
| 19 | 2026 | NTIRE 2026 Challenge on Efficient Low Light Image Enhancement: Methods and Results | 竞赛 | 中 | MobileIE 作为高效 LLIE 基线 |
| 20 | 2026 | A compact and interpretable multi-source framework for heterogeneous medical image classification | 医学 | 低 | 借用 4.2K/924M 规模类比 |
| 21 | 2026 | YOLOv11-SMS: An Improved Algorithm for Impurity Detection in Seed Cotton | 检测 | 低 | 无关 |
| 22 | 2026 | Multi-Type Ship Detection in Complex Marine Backgrounds Using an Enhanced YOLO-Based Network | 检测 | 低 | 无关 |
| 23 | 2026 | Bridging the Training-Deployment Gap: Gated Encoding and Multi-Scale Refinement for Efficient Quantization-Aware Image Enhancement | 部署 | 中 | 量化感知、训练-部署差距 |
| 24 | 2026 | FastSHADE: Fast Self-augmented Hierarchical Asymmetric Denoising for Efficient inference on mobile devices | 部署 | 中 | 移动端实时去噪 |
| 25 | 2026 | Advancing Visual Reliability: Color-Accurate Underwater Image Enhancement for Real-Time Underwater Missions | 水下 | 中 | 实时 UIE |
| 26 | 2026 | Revisiting Lightweight Low-Light Image Enhancement: From a YUV Color Space Perspective | 同组接续 | **高** | 同第一作者 Hailong Yan；单图 YUV 轻量 LLIE |
| 27 | 2026 | Escalator Falling Behavior Detection Algorithm Based on YOLOv5-MSPP | 检测 | 低 | 无关 |
| 28 | 2026 | ViMAF-YOLO: a lightweight detection model integrating multi-scale feature fusion and attention mechanisms | 检测 | 低 | 无关 |
| 29 | 2025 | JND-Guided Light-Weight Neural Pre-Filter for Perceptual Image Coding | 感知 | 中 | JND 阈值/可见性，可用于时序损失加权 |
| 30 | 2025 | StereoAdapter: Adapting Stereo Depth Estimation to Underwater Scenes | 水下 | 低 | 无关 |
| 31 | 2025 | Towards Lightest Low-Light Image Enhancement Architecture for Mobile Devices（LiteIE） | 同组接续 | **高** | Hailong Yan 参与；超轻无监督单图 LLIE |
| 32 | 2025 | IGDNet: Zero-Shot Robust Underexposed Image Enhancement via Illumination-Guided and Denoising | 同组接续 | **高** | 同第一作者；零样本单图 LLIE |
| 33 | 2025 | A Two-Stage Exposure Controllable Network for Image Enhancement With Low-Light Degradation | 低光增强 | 中 | 可控曝光，图像 |

## 2. 分类统计

| 类别 | 数量 | 说明 |
| --- | --- | --- |
| 同组接续 | 5 | #13、#18、#26、#31、#32 |
| 视频/时序 | 1 | #14（AnyMod-LLVE）；#10 是视频管线但不是时序方法 |
| 竞赛/基线 | 3 | #1、#2、#19 |
| 轻量部署 | 3 | #12、#23、#24 |
| 低光增强 | 3 | #7、#16、#33 |
| 检测 | 7 | 与本题无关，占引用最大比例 |
| 医学/分割/红外/水下/其他 | 11 | 多为工程借用，无时序内容 |

## 3. 关键词扫描结果

对 33 条标题+摘要扫描 `video / temporal / flicker / frame / sequence / recurrent / consistency`：

| # | 命中词 | 实际情况 |
| --- | --- | --- |
| 3 | video, temporal, consistency | 图像生成中的“伪视频”特征传输，不是视频增强 |
| 6 | temporal | ISP 奖励流，与视频无关 |
| 12 | frame | 边缘 NPU 部署测量，非视频帧一致性 |
| 14 | video | **唯一的视频低光增强** |
| 30 | recurrent | 立体深度网络结构，无关 |
| 31 | consistency | 指无监督损失一致性，非时序 |

4 条无摘要（#5、#11、#15、#16），按标题判断与视频/时序无关。

## 4. 结论

1. **没有“MobileIE + 视频闪烁/时序一致性”的直接工作。** 33 条引用中，唯一涉及视频低光增强的是 AnyMod-LLVE，但它走多模态+大模型路线，与“4K 参数单图网络扩展”不构成重复。
2. **同组最可能“往后做了一点”的是 #26（YUV）与 #31（LiteIE）/ #32（IGDNet）**，但三篇都是单图任务；#13 是水下图像；#18 是重参数化图像复原。它们构成“方法邻近风险”，不是“问题重复”。
3. **竞赛报告（#1、#2、#19）说明 MobileIE 已成为端侧效率基线**，未来投稿需要同时报告参数、延迟、能耗，不能只报 PSNR。
4. **#29 的 JND 思路值得吸收**：把“人眼可见的闪烁”纳入损失权重，是与现有 temporal loss 形成差异化的一个抓手。

> 注：S2 的 33 条与 OpenAlex 的 18 条不完全重合；完整清单以 `data/` 原始 JSON 为准。
