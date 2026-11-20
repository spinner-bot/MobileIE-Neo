# 单向传输与碰撞信息的先例边界

2026-11-20；Codex  (sqs_A100)。P1已关闭，不解封R51/R73/R74/R84/R86/R88或T1/T2的局部网格。

[Softmax Splatting, CVPR2020](https://arxiv.org/html/2003.05534v1)实际读摘要、§3.1部分及importance metric文字。前向传输、归一化、碰撞选择、颜色/特征传输和端到端学习均已有先例，不因本项目改成因果就成为原创。

[OCAI, CVPR2024](https://arxiv.org/html/2403.18092v1)实际读§3.1的occlusion weighting/公式10–11及hole-filling开头。它使用FB一致性产生遮挡权重，并非已经证明单向field足以验证正确性。[Splatting-based Synthesis, WACV2023](https://arxiv.org/abs/2201.10075)本次只读作者摘要和CVF检索提供的公式2–4/方法摘录：除光度、FB一致性外，源坐标3×3 Gaussian下flow方差也已用于merge metric。未声称完整正文或源码复现；本轮CVF PDF直接打开403，HTML备用入口成功的论文另行注明。

一个尚未验证的具体问题：用单向field在**目标坐标的实际贡献者集合**上累计位移均值/二阶矩，能否以更少field构建有用的历史接受域？它不同于固定源窗口的flow方差，但这一差别不是已成立的新颖性。常量平移的贡献者位移方差为零；不同位移源的碰撞可被读出；任意错误但一一映射的field仍可能方差零，因此不能称其几何真值、校准概率或FB等价替代。

先只做冻结原F0、两个既定训练批/双seed的有限筛查；加入“只估backward、取消FB”的直接廉价控制，避免把省掉一次calc本身包装为新方法。原FB、无拒绝push、目标贡献者方差push使用同已计算fields，GT-derived运动只用于所有输出共同评分。若缺乏质量—时间独立信号，停止该固定算子，不训练/调阈值或另扫传统方法。calc次数减少不等于完整流水线或手机加速。

旁路检索回到[PiLLVD原文](https://arxiv.org/html/2611.16515v1)摘要，其来源此前note38已登记Restricted，本轮未下载/查看像素，不能当新完成数据。没有将搜索结果中消费相机或无关3DGS页面作为技术依据。
