# 探索阶段文献地图：机制、组合与重复风险

- 创建：2026-11-10 18:16 +08:00 · Codex  (sqs_A100)
- 触发：自主前期调研，与4号工作预实验共同筛选多种方法及组合。
- 检索：官方CVF、arXiv、IPOL；未使用搜索摘要中的排名作为性能结论。
- 阅读层级：多数为官方摘要/方法概述核查；MobileIE另对照了本地代码与论文相关章节，FastLLVE/CRISP进一步阅读了方法及训练章节。尚未全部精读、复现，不作“领域空白”判断。
- 关联：[第一轮探索协议](../../4.temporal-consistency/docs/exploration-r1-protocol.md)。

## 已核实的机制入口

| 工作与一手来源 | 本轮核查到的机制 | 对本项目的用途与风险 |
| --- | --- | --- |
| [StableLLVE，CVPR2021](https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_Learning_Temporal_Consistency_for_Low_Light_Video_Enhancement_From_Single_CVPR_2021_paper.html) | 从静态图像合成运动短序列并施加一致性 | 静态图训练视频稳定性已有直接先例，不能仅以“合成帧+时序损失”为创新 |
| [FastLLVE](https://arxiv.org/abs/2308.06749) | 强度相关LUT与视频亮度一致性 | 增强映射的时序建模有相关工作，需比较映射参数化与计算成本 |
| [Deep Video Prior](https://arxiv.org/abs/2010.11838) | 在单段原始/处理视频上优化一致性网络 | 通用后处理对照；每视频优化成本必须报告，不能混作零训练成本在线方法 |
| [VBM3D实现与变体，IPOL2021](https://www.ipol.im/pub/art/2021/340/) | 相似块分组和跨帧变换域去噪 | 传统跨帧冗余的强控制；高斯噪声模型和泊松/JPEG输入的适配需核查 |
| [BRVE，CVPR2024](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Binarized_Low-light_Raw_Video_Enhancement_CVPR_2024_paper.html) | 时空shift与分布感知二值卷积 | 极轻量时间信息聚合并非空白；RAW任务与本项目RGB输入不同，不能直接混表 |
| [RetinexMCNet，ICCV2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_RetinexMCNet_A_Memory_Controller_Dominated_Network_for_Low-Light_Video_Enhancement_ICCV_2025_paper.pdf) | 单帧增强与记忆控制分阶段，结合Retinex分解 | “分解+记忆”的组合已有强相关先例；需要核查其更新方式和资源开销 |
| [TempRetinex，v2](https://arxiv.org/abs/2511.09609v2) | 亮度一致性预处理、多尺度时间损失、遮挡mask、反向推理与集成 | 组合训练/推理策略的实例；不能把预处理或遮挡mask本身当新贡献 |
| [DWTA-Net，v2](https://arxiv.org/abs/2510.09450v2) | 多帧增强后循环去噪，光流引导动态聚合和纹理自适应损失 | “增强+去噪+可靠性/纹理”有直接重叠，需寻找受限预算下更具体的问题 |
| [VLLVE/VLLVE++，2026预印本](https://arxiv.org/abs/2602.08699v1) | 视角相关/无关分解与对应关系联合优化 | 泛称“光照/内容分开”“退化与对齐共同优化”不足以建立差异 |
| [CRISP，2026预印本](https://arxiv.org/abs/2610.24299v1) | 两帧循环架构与渐进亮度监督，可选语义先验 | 课程式训练有近期直接相关工作；可研究低成本训练调度，但需区分贡献 |

预印本身份与已发表工作分别标注；本表并非穷尽清单。TempRetinex和DWTA当前v2的标题、作者、术语相对初版有变，以上以直接打开的arXiv页为准。

## 调研驱动的预实验

1. 先用普通去噪、输出EMA、低光光流和真值运动补偿判断瓶颈，避免一开始引入重模型。
2. 如果传统预处理伤害画质，研究增强与降噪的联合适配，而不是直接把滤波器堆到输入。
3. 如果真值对齐有明显收益而估计光流弱，比较可靠性建模、低分辨率映射状态、轻量时空特征与无需显式光流的候选。
4. 如果单帧时序损失有效，仍需与已有静态图合成训练方案、普通噪声增广和去噪控制比较。
5. 组合研究应保留单项与交互效应。无互补收益、收益来自参数增加或未来帧信息的组合要明确记录。

## 需要深入阅读与复现的顺序

本轮进一步阅读的两篇：

- [FastLLVE方法章节](https://arxiv.org/html/2308.06749v1)：视频编码生成共享LUT及强度图，再进行像素映射，可附加去噪。它并非仅对单帧预测一个颜色表；比较因果延迟时还需核对视频窗口。对本项目的启示是把映射稳定与噪声恢复分别测量；其可组合设计已有先例，不能直接作为新颖性。
- [CRISP方法与课程章节](https://arxiv.org/html/2610.24299v1)：分别编码当前输入与历史输出后再对齐融合；训练逐渐接近完整目标，再加入时序约束。它说明历史状态可能传播恢复误差，联合结构与训练值得研究。本轮仅核查这些章节，不复制其性能排名为本项目结论。

StableLLVE官方HTML摘要可访问，但进一步打开PDF收到403；保留此访问失败，后续从作者代码/论文镜像补读，当前不将其方法细节标为已精读。

- 第一优先：MobileIE的训练与IWO；StableLLVE的合成运动/损失细节；FastLLVE的亮度一致性建模。
- 随预实验推进：DWTA/RetinexMCNet的对齐和状态更新；BRVE的时空shift；VBM3D的噪声适配。
- 作为路线风险核查：TempRetinex、VLLVE++、CRISP；先核对全文/代码，再考虑作为数值基线。

## 检索过程

- `video low light enhancement temporal consistency StableLLVE single images 2021`
- `real time low light video enhancement BRVE binary`
- `low light video enhancement efficient recurrent illumination temporal 2025 2026`
- 对arXiv 2610.24299、2510.09450、2602.08699、2511.09609逐一打开主页，核对版本及摘要。
- 过程性证据清单见[`data/exploration_r1_sources.json`](../data/exploration_r1_sources.json)，不将论文全文或检索页面整页复制入库。

## 修订记录

- 2026-11-10 18:16 +08:00 · Codex  (sqs_A100)：建立10项机制与风险地图；按阅读深度标记，不预设唯一方法或组合。
- 2026-11-10 18:26 +08:00 · Codex  (sqs_A100)：补读FastLLVE/CRISP方法与训练章节，记录StableLLVE PDF访问失败和未完成项。
