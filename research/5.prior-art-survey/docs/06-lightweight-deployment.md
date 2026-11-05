# 轻量部署与端侧效率：对 MobileIE 课题的启示

- 创建：2026-11-05 · claude code (local)
- 范围：重参数化、量化、蒸馏、端侧评测口径，以及它们与“时序闪烁”的交叉。

## 1. MobileIE 的效率设计

| 设计 | 作用 | 推理代价 |
| --- | --- | --- |
| MBRConv（多分支重参数化卷积） | 训练时 5×5/3×3/1×1/十字分支并行，推理合并为单卷积 | 0（合并后） |
| FST（特征自变换） | 二次融合，增强高频细节 | 小 |
| HDPA（分层双路径注意力） | 全局通道注意力 + 局部注意力 | 极小 |
| IWO（增量权重优化） | warm-up 阶段冻结/增量更新，改善收敛 | 0（仅训练） |
| LVW（局部方差加权损失） | 抑制过曝/噪声放大 | 0（仅训练） |

公开指标：约 4K 参数、0.924 GMACs、GPU 1,100+ FPS、手机 SoC 100+ FPS（论文口径）。

## 2. 同组/相邻的效率工作

| 工作 | 效率亮点 | 对我们意味着什么 |
| --- | --- | --- |
| **LiteIE**（ESWA 2025） | 58 参数；骁龙 8 Gen 3 上 4K 30 FPS；无监督 | “更轻”已被占位，不能只比参数量 |
| **FGDPA**（arXiv 2606.30314） | 4.23K 参数、600+ FPS；MBRConv-DCT + 双路径注意力；重参数化后零额外推理 | 重参数化 + 新模块的组合已被同组使用 |
| **DRNet**（TMM 2026） | 动态重参数化，五任务 all-in-one，参数效率高 | “动态重参数化”不宜作为主 novelty |
| **FastSHADE**（arXiv 2604.10275） | 移动 GPU 去噪，<50 ms（Adreno 840） | 端侧延迟/画质权衡的强基线 |
| **QAT 图像增强**（arXiv 2604.21743） | 量化感知训练，缓解训练-部署差距 | 量化是部署必需环节，需纳入评测 |
| **边缘 NPU 量化研究**（AIoTC 2026） | 强调延迟/能耗，而非 params/FLOPs | 我们的效率表应包含延迟/功耗 |

## 3. 量化与时序闪烁（重要新线索）

低比特量化会让**逐帧量化误差不同**，从而产生闪烁；即使单帧 PSNR 正常，视频也会抖。已有工作（主要是视频超分）开始专门处理：

| 工作 | 机制 | 指标/结果 |
| --- | --- | --- |
| **TAQ**（ECCV 2026，[ECCV](https://eccv.ecva.net/virtual/2026/poster/5539)、[paper note](https://en.papernotes.org/ECCV2026/vlm_efficiency/taq_static-deployable_temporal-aware_quantization_for-real-world-video-super-res/#ablation-study)） | 静态可部署的时序感知 PTQ：只学习 clipping bounds，约束量化模型与浮点模型的**帧差**一致 | LPIPS/TLPIPS 改善；Jetson Orin Nano INT8 最高 3.56× |
| **QBasicVSR**（NeurIPS 2025，[OpenReview](https://openreview.net/pdf?id=6oTJCnTkUA)） | 时序感知比特分配（FG-VBA/TS-LBA）+ 教师蒸馏，惩罚时序不一致 | VSR 低比特 SOTA |
| **PMQ-VE**（NeurIPS 2025，[PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/bcfdaf04b54a69f47623c973c864ee8d-Paper-Conference.pdf)） | 多帧视频增强的逐帧粗到细量化 | VSR/VFI/STVSR 的 SOTA |

**空白**：上述工作集中在视频超分/多帧网络；**没有工作研究“4K 参数级单图增强网络在 INT8/TFLite 部署后的时序闪烁”**。MobileIE 的最终目标是手机端，量化几乎不可避免，因此：

- 一个低成本的必做实验：把 `lolv1_best_slim.pkl` 转 ONNX → INT8/TFLite，测量化前后的闪烁（同一视频、同一指标）；
- 如果量化显著放大闪烁，**“时序感知量化/部署”可以成为第二个贡献点**，而且与主线（训练期时序约束）天然互补；
- 需要验证量化后的 `slim` 结构是否仍保持 4K 参数与延迟优势。

## 4. 评测口径建议

| 指标 | 为什么必须报告 |
| --- | --- |
| 参数量 | 与 MobileIE/LiteIE/FGDPA 对齐 |
| MACs/FLOPs | 论文常用，但不等于延迟 |
| GPU 延迟（batch=1，400×600，warmup） | 与 MobileIE 论文口径对齐 |
| 手机/TFLite 延迟、FPS | 目标部署平台 |
| 峰值内存/功耗（若可得） | 端侧硬约束 |
| 量化前后闪烁 | 新增，体现部署真实性 |

**不要只报 FPS**：分辨率、batch、设备、是否包含预处理都会影响；Edge NPU 研究（#12）明确指出 params/FLOPs 不能代表实际延迟/能耗。

## 5. 对 4 号工作的建议

1. 主线方法保持推理图不变（训练期时序损失），天然兼容量化；
2. P1 阶段增加“量化前后闪烁”实验，作为部署真实性的一部分；
3. 若后续加时序模块，优先可重参数化/可折叠结构（参考 FGDPA/NanoVSR），并测 INT8 后的时序稳定性；
4. 论文效率表同时给 GPU 与手机/TFLite 数据；至少给出量化方案与精度损失。

## 6. 待核实

- [ ] MobileIE 是否有官方 TFLite 模型与手机测速脚本（仓库有 `test_TFLite_RGB.py`、`torch_to_onnx.py`）；
- [ ] INT8 量化后 4K 参数网络的精度/延迟变化；
- [ ] TAQ/QBasicVSR 的 TLPIPS 定义能否直接用于我们的闪烁指标；
- [ ] 是否有工作专门研究“图像增强模型量化的时序闪烁”（目前未检索到，需 Google Scholar 复核）。
