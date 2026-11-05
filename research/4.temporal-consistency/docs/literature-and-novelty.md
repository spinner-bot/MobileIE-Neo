# 文献与新颖性核查（MobileIE 视频时序一致性）

- 创建：2026-11-05 · claude code (local)
- 触发：4 号工作启动，需在动模型前确认“是否已有人解决 MobileIE 的视频闪烁”。
- 后续：更完整的引用全表、同组接续与 TeleAI 核查见 [5 号前期调研](../../5.prior-art-survey/README.md)；本文保留为 4 号工作的简要定位说明。

## 1. 检索方法

| 渠道 | 做法 | 结果 |
| --- | --- | --- |
| Semantic Scholar API | 查询 arXiv `2507.01838` 的引用列表（`/citations`） | 返回了约 20 条引用；后续请求触发 429 限流，**列表不完整** |
| Web 检索 | `MobileIE video temporal consistency`、`video low-light enhancement temporal consistency lightweight`、`flicker suppression` 等 | 未发现直接研究 MobileIE 视频闪烁的工作 |
| arXiv / CVF / ACM 页面 | 对最相关条目核对标题、年份、方法要点 | 见下表 |

**局限**：Google Scholar 未做逐条人工核查；Semantic Scholar 限流导致引用列表可能遗漏；2026 年的新工作较多，结论需在正式开题前用 Google Scholar 再复核一次。

## 2. MobileIE 的引用情况

在检索到的引用中，**没有发现以“MobileIE + 视频/时序一致性/闪烁”为主题的工作**。相关引用主要分三类：

| 类型 | 代表工作 | 与我们的关系 |
| --- | --- | --- |
| 作为轻量 baseline / 竞赛方案 | *Learned Smartphone ISP on Mobile GPUs: Mobile AI 2026 Challenge Report*（arXiv 2611.05008）；*Efficient Image Denoising on Smartphone GPUs: Mobile AI 2026 Challenge Report*（arXiv 2611.00974）；*NTIRE 2026 Challenge on Efficient Low Light Image Enhancement*（arXiv 2605.02212） | 说明 MobileIE 已成为端侧效率基线，但均按**单图**使用 |
| 图像增强 / 下游应用 | *Real-Time Underwater Image Enhancement via Frequency-Guided Dual-Path Attention*（arXiv 2606.30314）；*AI-Driven Illumination-Adaptive Enhancement for Robust Mobile AR Games*（DOI 10.1145/3841358.3841415）；*Zero-Shot Illumination and Noise Estimation for Edge-Deployable LLIE*（IoT-J 2026） | 沿用单图增强，不处理帧间一致性 |
| 视频低光增强（仅 1 篇） | *AnyMod-LLVE: Low-Light Video Enhancement with Modality-Agnostic Inference*（ICML 2026，arXiv 2606.11186） | 多模态、重预训练的视频 LLVE，不是轻量端侧方案；仅在引用层面相关 |

其余引用（医学影像、焊接、CT、苹果叶病、人体身高、红外等）与本题无关。

## 3. 相邻工作对比

| 工作 | 场景 | 时序机制 | 效率/部署 | 对我们的意义 |
| --- | --- | --- | --- | --- |
| **UbiEnlight**（ACM 2026，[DOI](https://dl.acm.org/doi/10.1145/3832024)） | 手机/智能眼镜实时低光视频 | 频谱先验 + spectrum-gated temporal reuse + 相关噪声 + 相位正则，**不用光流** | 扩散 Transformer；报告端到端 18.81 s → 60 ms、时序一致性提升 54.76% | 代表“重模型 + 时序复用”路线；我们的差异是 4K 参数 CNN、零/极低推理开销 |
| **Efficient Real-Time Raw-to-Raw Denoising for Extreme Low-Light UHD Video on Mobile**（CVPR 2026，[CVF](https://openaccess.thecvf.com/content/CVPR2026/papers/Pochimireddy_Efficient_Real-Time_Raw-to-Raw_Denoising_for_Extreme_Low-Light_Ultra_HD_Video_CVPR_2026_paper.pdf)） | 手机 RAW 视频（4K/8K，1 lux） | 时序降噪 + 运动补偿，强约束 33 ms / ~250 mA | RAW 域、ISP 管线、重参数化/量化 | 证明“端侧实时 + 时序一致性”是活跃赛道；我们做 RGB 域、4K 参数 |
| **AnyMod-LLVE**（ICML 2026，[arXiv](https://arxiv.org/abs/2606.11186)） | 视频低光增强，模态可缺失 | 空间-频谱双门控、多模态预训练 | 模型重、依赖大规模预训练 | 说明视频 LLVE 主流仍是重模型；不是直接竞品 |
| **LIVE-Net**（Signal, Image and Video Processing 2026，[链接](https://jx.nstl.gov.cn/paper_detail.html?id=60a0b3d5fbf47fff96d4dc0d690e0473)） | 图像 + 视频低光增强统一网络 | 时序一致性约束 | 轻量网络 | **最接近的轻量竞品，需精读全文核对参数量/延迟** |
| **ViD-EnlightenGAN**（IEEE IoT-J 2026，[IEEE](https://ieeexplore.ieee.org/document/11556131)） | 无监督低光视频增强 | 时序注意力 + 多判别器，无光流 | 面向 IoT 边缘 | 与我们的“无光流”思路相近；需比较其模型规模 |
| **EeveeDark**（RA-L 2026，[arXiv](https://arxiv.org/html/2607.06217)） | 低光视频（RAW + 事件） | 事件引导、二值网络 | 机器人/嵌入式 | 多模态路线，与 RGB 单目不同 |
| **BinRVR**（TPAMI 2026，[arXiv](https://arxiv.org/abs/2608.16756)） | RAW 视频复原 | 二值化时空交互模块 | 参数/计算降约 96% | 说明“极致压缩 + 时空建模”可行；可借鉴其效率评估口径 |
| **BlazeBVD**（ECCV 2024，[ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02526.pdf)） | 盲视频去闪烁 | 直方图先验 + 全局/局部去闪烁模块 | 比 SOTA 快约 10× | 后处理路线的强基线，需纳入对照 |
| **NanoVSR**（arXiv 2607.10495，[alphaXiv](https://www.alphaxiv.org/abs/2607.10495)） | 边缘端视频超分 | 多分支训练 + 结构重参数化，推理单分支；无显式光流 | 226K 参数 / 43.8 FPS（Jetson Orin NX） | **与我们的“训练复杂、推理极简”思路最契合**，可作为方法学参考 |
| **Zero-DCE（视频版）**（[CityU PDF](https://scholars.cityu.edu.hk/files/182761625/132903062.pdf)） | 低光视频增强 | 曝光一致性 + 颜色一致性损失；光流仅在训练期使用 | 推理仅曲线映射 | **方案 A（训练期时序损失、推理零开销）的直接先例** |
| **JFFRA**（arXiv 2505.16434，[ar5iv](https://ar5iv.labs.arxiv.org/html/2505.16434)） | 视频复原 | 遮挡感知的时序损失 + 光流/特征联合细化 | 较重 | 时序损失设计可借鉴（遮挡掩码） |

> 说明：FastLLVE、StableLLVE、AdaEnlight 等在检索摘要中出现，但未拿到可靠原文链接，暂不列入正式对比，后续补读。

## 4. 新颖性定位（初稿）

**空白**：MobileIE 是 4K 参数、0.924 GMACs 的单图增强网络，目前没有任何工作把它扩展到视频并处理帧间闪烁。

**候选卖点**：

> 面向极轻量单图增强网络的**零/极低推理开销时序一致性**：训练期引入时序约束（必要时配合可重参数化的轻量帧间分支），推理端仍保持单分支、4K 参数级部署，并在同等延迟下优于 EMA/光流后处理。

与相邻工作的差异：

1. **vs 重模型视频 LLVE**（UbiEnlight、AnyMod-LLVE）：我们不做扩散/多模态，目标是手机实时；
2. **vs RAW 域方案**（CVPR 2026、BinRVR）：我们做 RGB 域、与现有 MobileIE 权重/部署链兼容；
3. **vs 轻量视频网络**（LIVE-Net、ViD-EnlightenGAN）：我们的 backbone 只有 4K 参数，时序开销必须接近零；
4. **vs 后处理**（EMA、光流 EMA、BlazeBVD）：必须给出快速运动/遮挡/延迟下的边界，证明训练侧约束的必要性。

**必须诚实面对的风险**：3 号工作的光流 EMA 已把闪烁降低 73% 且画质不降。如果训练侧方案不能在“快速运动、无拖影、低延迟”上明显优于后处理，论文价值会被质疑。评价协议中必须把这三项作为主指标。

## 5. 对方案选型的启示

1. **优先方案 A：训练期时序一致性损失**——Zero-DCE 视频版、NanoVSR 都证明“训练期复杂、推理期简单”可行，且与 MobileIE 的重参数化哲学一致；
2. **若必须加模块**：参考 NanoVSR 的结构重参数化，让训练期的时序分支在推理时折叠掉；否则参数/延迟预算很难守住；
3. **时序损失需要运动处理**：训练期可用光流 warp 构造对齐监督（推理不用光流），并参考 JFFRA 的遮挡掩码避免错误对齐；
4. **必须对照后处理**：EMA、光流 EMA、BlazeBVD 风格的后处理，至少在慢速平移场景作为强基线。

## 6. 待办

- [ ] 用 Google Scholar 完整导出 MobileIE 引用列表，逐条核对（当前 S2 列表不完整）；
- [ ] 精读 LIVE-Net、ViD-EnlightenGAN、NanoVSR、Zero-DCE 视频版，记录参数量/延迟/时序指标口径；
- [ ] 核对 UbiEnlight 与 CVPR 2026 RAW 方案的时序指标定义，尽量复用其评价方式；
- [ ] 确认公开视频数据集的可获取性（SDSD、REDS、SID 等）与许可；
- [ ] 与导师确认“后处理够用”的判断边界，再决定是否把方案 A 作为主线。
