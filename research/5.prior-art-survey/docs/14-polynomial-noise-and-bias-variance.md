# 多项式噪声传播与偏差—方差诊断的先例

创建：2026-11-11T16:48:39.185995+08:00；作者：Codex  (sqs_A100)；原因：R6需明确根部算子矩分析的已知部分和适用假设。

[FC-AIDE，ICCV2019，arXiv v4正文](https://arxiv.org/html/1807.07569)：本次阅读§3.1–3.2及方法概述。它以不包含当前像素的上下文预测仿射/二次映射系数，推导无偏MSE风险估计。假设独立、零均值、对称噪声；二次项的结论使用对称性。多项式映射及噪声统计风险校正已有直接先例。

[Self-Calibrated VST，ICCV2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Herbreteau_Self-Calibrated_Variance-Stabilizing_Transformations_for_Real-World_Image_Denoising_ICCV_2025_paper.pdf)：仅检索到正文§3.3片段，讨论可学习变换、固定Gaussian去噪器及blind-spot自监督；未核验完整实现及视频性能。不能将“噪声方差适配变换”泛称新颖。

[Data-dependent Nonlinearity Analysis in CT Denoising CNNs](https://pmc.ncbi.nlm.nih.gov/articles/PMC9119294/)与[Tunable neural networks for CT image formation](https://pmc.ncbi.nlm.nih.gov/articles/PMC10157542/)：检索片段涉及重复噪声、局部线性噪声传播和偏差/协方差加权目标。直接正文访问受到验证页限制，未完成精读。这些是不同成像领域的分析先例，不能当作低光视频效果基线。

## 本项目的边界（推断）

FST的Jensen偏移与四阶中心矩分解是基础代数。它们可以检验某种机理解释是否成立，不能单独形成论文创新。MobileIE的head包含PReLU且body接收非线性特征，JPEG/量化/截断也会带来非零均值和相关性，不能套用像素独立对称噪声的风险证明。

减去条件方差对应的常数只改变该层均值；下游RGB波动是否改变需要实际干预。R6将校准/评价随机退化分开，并将理想统计参考与可部署算法分开记录。只有在oracle作用明确、有可估计统计与成本优势、且与已有噪声感知/多项式方法区分清楚时，才考虑进一步设计。

修改：2026-11-11T22:40:33.143257+08:00；作者：Codex  (sqs_A100)；原因：Noise2VST由片段检索升级为正文v2/固定源码阅读，保留以上历史阅读状态。

后续已读§3.1–3.4/4.1及数据假设/效率，固定官方实现02b839d，详见[21](21-noise-transforms-and-resampling-assumptions.md)；并未运行其模型或确认视频有效性。R22是空间插值矩，不是噪声估计。
