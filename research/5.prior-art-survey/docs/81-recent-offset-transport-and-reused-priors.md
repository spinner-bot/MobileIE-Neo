# 81｜近期偏移传输与已有训练/滤波先例的归并

2026-11-21；Codex  (sqs_A100)。A1负结项后定向补读；没有下载/执行外部模型、代码、权重或像素。

新增实际读取[Fast Training-Free LLVE，arXiv2611.22551v1](https://arxiv.org/html/2611.22551v1)摘要、§3.3–3.4及§5.3/5.6片段：先对低光输入作因果运动补偿递归融合，再在非关键帧复用关键帧的中间扩散增强偏移；当前帧仍提供自身编码、注意力和skip，并非复制整个历史状态。采用固定关键帧调度，仍需flow、VAE及多次网络运算；作者明确披露低分辨率感知退化和离线速度。这里不导入作者成绩为本项目数据，也不称已精读全部证明/附录或完成复现。

它进一步限制“先滤输入＋只传增强偏移＋关键帧复用”的宽泛创新叙事。MobileIE前馈增强与扩散反演不同，A1是当前低光上应用局部RGB仿射场，没有其当前编码/skip或扩散过程；不能据该文正例挽救A1固定拟合，也不能据A1负例否定该文。基础残差恒等与当前噪声回注见[note44](44-mapping-state-and-factor-identifiability.md)。

本轮回读CRISP v1方法概述、F2R v1 §3.1和FilterFusion v1 §III，但它们**不是新发现**：CRISP渐进目标已在note08/13；FilterFusion低分辨率控制/方差/滤波合成已在[note53](53-variance-state-and-filter-synthesis-priors.md)；F2R v2及固定作者代码对未来帧、中心coarse与flow的实际依赖已在[note66](66-kernel-mixtures-and-reference-preserving-learning.md)。本轮较早v1正文不能覆盖此前更细代码核查，尤其不将“屏蔽中心联合输入”扩写成完全不读中心或无flow。

[SAT正式发表页](https://www.nature.com/articles/s41598-026-38431-9)只读摘要/引言，支持时间/空间shift与隐式对齐已有先例，未精读完整方法/实验；一次find重定向认证失败后未绕过。[Efficient recurrent real video restoration出版页](https://www.sciencedirect.com/science/article/pii/S1051200424004767)403，只有搜索摘要线索，不作为算法依据。没有把旁路扩散生成/新闻或第三方总结当技术证据，也不据检索没找到同名方法声称新颖性。

归并结论：一般循环、课程、盲时间锚、映射/偏移平滑、低成本权重混合都已有直接先例。下一候选必须交代具体可用信息和推理预算，并先于训练设计能区分空间修正/历史信息/额外未来或计算的控制；不能以一般组件组合或新哈希数量当论文贡献。所有旧STOP与独立最终封存保持。
