# 16｜优化与重参数化控制

创建：2026-11-11T18:39:15.432022+08:00；作者：Codex  (sqs_A100)；原因：R8可折叠copy与R9普通学习率增益要求明确已有工作及训练等价边界。

RepVGG已将训练多分支折叠为单卷积推理，不能把可折叠并行分支称为视频创新。本次核查[CVPR2021作者页面与摘要](https://openaccess.thecvf.com/content/CVPR2021/html/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.html)；PDF直接打开403，仅检索摘录可见，未声称本次完整精读。

[RepOptimizers，arXiv2205.15242v4](https://arxiv.org/html/2205.15242v4)本次阅读§3.1–3.4及§4.2：在特定线性分支/常数尺度条件下，以匹配初始化和更新规则建立CSLA与梯度重参数化的等价，而非仅凭推理可合并就推导训练等价。[作者仓库README](https://github.com/DingXiaoH/RepOptimizers)标ICLR2023，说明SGD/AdamW实现并提示gradient clipping会破坏所示等价；本次没有运行官方代码或读取完整实现。

## 本项目推导与限制

同输入同padding卷积有W_eff=W_main+W_aux，R8copy推理函数族不扩大。对最简单无动量/无衰减/无clip的SGD，两分支同梯度g、各学习率eta_m/eta_a时，合并权重更新为−(eta_m+eta_a)g；这是线性恒等式，不是新颖性或本项目Adam实验的等价证明。

R8主干/分支分别clip并各自Adam历史，R9全主干统一clip，首层分组还更新bias；因此只称优化强对照，不能称复现RepOptimizer或精确相同训练轨迹。R9多数步被clip，未来任何等价声明需包含实际更新器、epsilon、状态、clip和有限精度。

R9全模型学习率改变已经产生比早期微小结构增量大得多的收益。本项目据此要求：结构、优化与实际历史贡献分别控制；重新训练候选应使用同一强优化条件，保留同权重current-history和可折叠空间控制。论文不能把“多分支训练后单分支部署”或“改优化获益”本身当新视频方法。

关联：[0018](../../../docs/进展报告/0018-T-26111118-输入时域化与优化强对照.md)、[0019](../../../docs/进展报告/0019-T-26111118-优化基线与新增开发集.md)。这些是本地探索的解释约束，不把图像分类结果外推为低光视频效果。
