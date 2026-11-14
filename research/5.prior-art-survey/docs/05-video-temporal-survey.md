# 视频时序一致性/去闪烁方法综述

> 2026-11-12核查补记（Codex  (sqs_A100)）：这是早期检索草案，含未核实分类/会议和空白判断，不能直接作论文事实。FastLLVE已核对为ACM MM2023，视频编码/LUT结构不能归为普通RNN；“没有工作把4K单图扩为视频”等句不作为已核验新颖性。实际阅读层级及最新边界以08、40、43、44号读记和进展报告为准。


- 创建：2026-11-05 · claude code (local)
- 范围：低光视频增强、视频复原、去闪烁、移动端实时增强中与“帧间一致性”相关的代表性工作。
- 目的：为 4 号工作的方法选型提供坐标，并明确我们的差异点。

## 1. 方法分类

| 类别 | 核心机制 | 代表工作 | 优点 | 缺点 | 对我们的意义 |
| --- | --- | --- | --- | --- | --- |
| A. 对齐 + 时序损失 | 训练期用光流 warp 相邻帧，加一致性损失；推理不用光流 | Zero-DCE 视频版、JFFRA、StableLLVE、BlazeBVD 的 warping loss | 推理零开销；实现简单 | 依赖训练期光流质量；遮挡/快速运动易错 | **方案 A 的直接依据** |
| B. 循环/状态网络 | 隐状态跨帧传播（RNN/convGRU/双向 recurrent） | FastLLVE、EvLight++、NanoVSR | 显式利用历史；效果好 | 引入帧缓冲/延迟；状态难压缩；训练不稳定 | 方案 B 的备选 |
| C. 时序注意力/特征复用（无光流） | 跨帧 attention、feature reuse、spectrum-gated reuse | UbiEnlight、ViD-EnlightenGAN | 避免光流；对运动有一定鲁棒性 | 模型通常较重；注意力开销大 | 若做轻量模块可借鉴“无光流”思路 |
| D. 后处理去闪烁 | 直方图先验、全局/局部去闪烁、滤波 | BlazeBVD、DeFlicker、EMA/光流 EMA | 与模型解耦；可直接用在 MobileIE 输出 | 拖影、延迟、对快速运动/遮挡失效 | **必须纳入对照** |
| E. 多模态/事件/RAW | 事件流、红外、RAW 域、ISP 管线 | AnyMod-LLVE、EeveeDark、CVPR 2026 Raw-to-Raw、BinRVR | 信息更充分；端侧 ISP 可落地 | 需要额外传感器/RAW；与 RGB 单目不兼容 | 说明赛道活跃，但不是我们的直接竞品 |
| F. 生成式/扩散 | 扩散先验、时序自注意力、相关噪声 | UbiEnlight、CS³-Diff、InstaVSR | 画质上限高 | 计算重、延迟高、难上手机 | 作为“重模型上限”参照 |

## 2. 代表工作明细

### 2.1 对齐 + 时序损失

- **Zero-DCE 视频版**：在 Zero-DCE 的曲线估计上增加曝光一致性、颜色一致性损失；光流只用于训练预处理。  
  [CityU PDF](https://scholars.cityu.edu.hk/files/182761625/132903062.pdf)  
  **启示**：证明“训练期时序损失 + 极轻推理”可行；但它是曲线模型，不是 4K 参数 CNN，也未处理局部高频噪声。
- **JFFRA**：光流与特征联合细化，遮挡感知时序损失。  
  [ar5iv](https://ar5iv.labs.arxiv.org/html/2505.16434)  
  **启示**：遮挡 mask 是必要的；可作为我们 loss 的设计参考。
- **StableLLVE / FastLLVE / AdaEnlight**：低光视频增强中的时序正则/轻量实时路线（检索中出现，原文待补读）。  
  **启示**：轻量视频 LLVE 已有竞争，需要核对参数量与延迟。
- **BlazeBVD**（ECCV 2024）：直方图辅助的盲视频去闪烁，全局+局部去闪烁模块，报告比 SOTA 快约 10×。  
  [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02526.pdf)  
  **启示**：后处理去闪烁的强基线；必须比较。

### 2.2 循环/状态网络

- **NanoVSR**（arXiv 2607.10495）：多分支训练 + 结构重参数化，推理单分支；无显式光流；226K 参数在 Jetson Orin NX 上 43.8 FPS。  
  [alphaXiv](https://www.alphaxiv.org/abs/2607.10495)  
  **启示**：与 MobileIE 的“训练复杂、推理极简”哲学一致；若做时序分支，应走可重参数化路线。
- **EvLight++**（arXiv 2408.16254）：事件相机低光视频增强，循环模块 + 时间损失。  
  [arXiv](https://arxiv.org/abs/2408.16254)  
  **启示**：多模态可获得更好时序，但依赖事件相机。
- **FastLLVE（ECCV 2024?）/ AdaEnlight**：轻量实时低光视频增强；具体指标待补读。

### 2.3 时序注意力/特征复用（无光流）

- **UbiEnlight**（ACM 2026）：手机/智能眼镜实时低光视频增强；频谱先验 + spectrum-gated temporal reuse + 相关噪声 + 相位正则，不用光流；报告端到端 18.81 s → 60 ms、时序一致性提升 54.76%。  
  [ACM](https://dl.acm.org/doi/10.1145/3832024)  
  **启示**：无光流时序复用是趋势；但其模型是扩散 Transformer，和 4K 参数 CNN 不在一个量级。
- **ViD-EnlightenGAN**（IEEE IoT-J 2026）：无监督 GAN + 时序注意力 + 多判别器，无光流。  
  [IEEE](https://ieeexplore.ieee.org/document/11556131)  
  **启示**：边缘设备上的时序 GAN；需比较模型规模。

### 2.4 后处理去闪烁

- **EMA / 光流 EMA**：我们在 3 号工作已测；慢速平移下闪烁 18.07 → 4.83（−73%），PSNR/SSIM 反升。  
  局限：快速运动拖影、遮挡失效、1 帧延迟。
- **BlazeBVD**：见 2.1；作为强后处理基线。
- **DeFlicker**：面向高速夜间/人工光源录制的实时去闪烁（广播帧率）。  
  [PDF](https://eccs.net/Vienna-prog.pdf)  
  **启示**：工程界已有实时去闪烁产品化思路，需要说明我们的场景差异（增强模型引入的闪烁，而非光源频闪）。

### 2.5 多模态 / RAW / ISP

- **AnyMod-LLVE**（ICML 2026）：模态可缺失的视频低光增强，AMNet + 空间-频谱双门控 + 大规模多模态预训练。  
  [arXiv](https://arxiv.org/abs/2606.11186)  
  **启示**：视频 LLVE 的主流是重模型；我们的 4K 参数路线是差异化，但不能在画质上硬碰。
- **CVPR 2026 Raw-to-Raw Denoising for Extreme Low-Light UHD Video**：手机 RAW 视频，33 ms / ~250 mA 约束，重参数化 + 量化 + ISP 管线。  
  [CVF](https://openaccess.thecvf.com/content/CVPR2026/papers/Pochimireddy_Efficient_Real-Time_Raw-to-Raw_Denoising_for_Extreme_Low-Light_Ultra_HD_Video_CVPR_2026_paper.pdf)  
  **启示**：端侧实时的评价标准应包含延迟/功耗；RAW 域不是我们的战场。
- **BinRVR**（TPAMI 2026）：二值化 RAW 视频复原，参数/计算降约 96%。  
  [arXiv](https://arxiv.org/abs/2608.16756)  
  **启示**：极致压缩 + 时空建模可行，但需要 RAW 数据。
- **EeveeDark**（RA-L 2026）：RAW + 事件的二值网络。  
  [arXiv](https://arxiv.org/html/2607.06217)  
  **启示**：多模态端侧路线。

## 3. 文献中的评价指标

| 指标 | 常见定义 | 问题 |
| --- | --- | --- |
| Temporal consistency / warping error | 用光流 warp 后相邻帧的 L1/L2 | 光流误差会污染指标 |
| PSNR / SSIM | 与 GT 比较 | 不反映时序 |
| Flicker score | 亮度/颜色直方图差、帧差 | 定义不统一 |
| LPIPS | 感知相似度 | 与时序无关 |
| Latency / FPS / power | 端侧部署 | 报告口径差异大（分辨率、batch、设备） |

**建议**：我们同时报告“已知运动对齐残差 + 光流对齐残差 + 逐像素时间 std + JND 超限比例 + 时间频谱”，并注明单位（0–255）与运动条件。这在现有文献中较少见，可作为方法之外的贡献。

## 4. 空白与机会

1. **极轻量单图 backbone 的时序扩展**：现有轻量视频方法多为专用视频网络（FastLLVE、LIVE-Net）或重模型（UbiEnlight、AnyMod-LLVE）；没有工作把 4K 参数单图增强器扩展到视频。
2. **零推理开销的训练期时序约束**：Zero-DCE 视频版有先例，但未用于 4K 参数 CNN，也未处理局部噪声放大。
3. **可见性驱动的时序损失**：JND/平坦区加权在去闪烁/编码中出现，但很少用于“增强模型引入的闪烁”。
4. **重参数化兼容**：FGDPA/DRNet 证明重参数化可零推理开销；把时序分支也做成可重参数化，是尚未被占位的组合。
5. **手机实时 + 真实夜景视频**：多数工作在 GPU 或离线视频上评测；端侧 30 FPS + 真实噪声的联合验证仍是稀缺资源。

## 5. 对 4 号工作的直接建议

- 主线仍是 **方案 A（训练期时序损失，推理零开销）**；
- 若需要结构模块，走 **可重参数化时序分支**（参考 NanoVSR、FGDPA）；
- 损失设计加入 **遮挡 mask + JND/平坦区加权**，形成与 Zero-DCE/JFFRA 的差异；
- 必须对照 **EMA / 光流 EMA / BlazeBVD 风格后处理**，并报告快速运动与延迟；
- 评测同时给 **闪烁、画质、参数、MACs、延迟、功耗（若可得）**，避免只报 FPS。

## 6. 待补读

- [ ] LIVE-Net 全文（参数量/延迟/时序指标）
- [ ] FastLLVE、StableLLVE、AdaEnlight 原文
- [ ] BlazeBVD 的快速运动表现
- [ ] UbiEnlight 的时序指标定义（是否可直接复用）
- [ ] CVPR 2026 Raw-to-Raw 的延迟/功耗测量方法
