# 蒸馏目标、局部状态支持与下一前置检验

2026-11-17；Codex  (sqs_A100)。R82完整收口后定向阅读与源码核查，不扩大基线家族或把关键词未搜到当原创。阅读范围见[source-reading001](../process/operator-r83/source-reading001.json)。

[2020 Optical Flow Distillation](https://arxiv.org/html/2007.05146v1)§3.2 Eq2–5已有输出KD以及有/无flow教师之差的残差KD，§3.3另有对齐低秩约束。这是note55已有先例回访。若两边减去相同冻结F0，残差L2与直接输出L2代数相同；“反事实创新量”命名不增加机制。[PocketDVDNet](https://arxiv.org/html/2601.16780)§2已有结构剪枝、域适配teacher与teacher/GT损失混合；648096参数、5帧patch和桌面GPU结果不等于本项目的因果/手机证据。未执行其代码或取得权重/新视频。

[Sobolev Training](https://arxiv.org/html/1706.04859)引言/Figure1文字已说明导数匹配与随机方向投影；[Jacobian Matching](https://arxiv.org/html/1803.00443)摘要/引言已有输入加噪与Jacobian蒸馏的联系。[TinyFlow](https://arxiv.org/html/2610.33338v1)III-H教师项比较quarter-resolution flow坐标。普通随机探针、算子/导数拟合或移除推理期teacher均不独立作为创新。这里没有完成穷尽查重或复现。

本地[R28勘误](../../4.temporal-consistency/docs/training-target-provenance-r83.md)很重要：其teacher提供对应场，实际重建target是GT，不能因R80旧文字把它误记为已经做过strong-rec输出蒸馏。历史冻结协议/结果保留，单独纠正推论。

## 不先训练另一个局部修正头

R82的H为1×1、depthwise5×5、1×1组合。固定当前c与参数，单步输出对给定历史RGB状态h只读半径2的邻域；减去H(c,c)不改变这一点。其整个递归状态可包含更早传播信息，不能把单步局部性说成整段视频只有2像素信息半径。R64/R74等有显式非局部采样，也不能套用这个限制否定它们。

强rec在固定当前c、flow、接纳mask时是`T(c,h)=c+.5*m*(B_f h−c)`，对h的线性部分A=.5*m*B_f可以读取邻域外位置。令反事实状态各坐标独立、零均值/单位方差，局部函数对未知邻域外系数的不可约均方项为相应系数平方和；随机探针匹配还满足E||D z||²=||D||F²。这只是条件方差/线性代数，不是新定理、真实RGB视频下界或R82失败的单一因果解释。真实状态相关，且若重估flow，完整教师已非这个固定线性算子。

下一次只在固定训练窗量化两个问题：原strong-rec状态算子有多少系数能量落在R82单步支持以外；R74式亮GT估计场与由可观察F0输入估计场的算子是否有实质差别。两者仍是估计对应，不把GT-derived称作真实flow。若没有足够差距，不以新术语继续投非局部算子蒸馏；若有，只支持下一不同核心原型的前置理由，不证明新颖性或质量收益，不换旧R74/R82的teacher/loss/核挽救。

检验不读新DEV或保留最终，不训练/安装新外部模型。所有正式设计仍需相同成本的空间/因果控制、独立历史作用、强后处理及完整质量—时间—成本门槛；普通KD和参数少不足以形成论文。
