# 79｜自由潜在状态与受限传输状态：先例及接续边界

创建：2026-11-21T10:47:42.154772+08:00；作者：Codex  (sqs_A100)。触发：O1完整门失败后归并旧状态实现，避免继续soft相关/teacher/probe网格。实际阅读范围见[L1来源记录](../../8.video-backbone/latent-core/process/source-reading001.json)，没有执行外部实现或下载新权重/视频。

[RLSP v1](https://arxiv.org/html/1909.08080v1)§3/3.1已有可学习潜在特征递归、输出反馈及shuffle，且输入明确包含下一帧，输出为Y另处理色度；不能把它称作已复现的零未来LLVE参考。[GRU-VD v1](https://arxiv.org/html/2210.09135v1)III-A–C/式1–5已有reset/candidate/update、已知噪声输入及输出递归。故一般自由状态/门控/无显式flow不是空白。

[NAFNet v1](https://arxiv.org/html/2204.04676v1)§4/式1–7已有乘法SimpleGate/SCA，不将再加乘法称新原理。[Turtle v1](https://arxiv.org/html/2410.03936v1)§3/3.1开头明确当前encoder、依历史decoder和历史队列；本轮未精读后续对齐/路由公式，不由此声称与其具体实现等价或取得其性能。

源码复核给出的局部区别：R64的quarter8状态仍是project(current)与warp(previous)的凸混合，mix卷积只用于field；R73是原head12通道的均值/二阶矩；O1存上一RGB融合及当前描述，不保存自由循环特征。ReBotNet现有任务适配传播增强RGB、另读两帧，不是独立自由latent state基线。它们已完成的负结果不能自动覆盖自由状态，但这一差别也不证明其有效或新颖。

下一L1只登记一个已知思想的具体任务基线：保留各seed原F0全部权重和当前增强路径起点，quarter32隐状态用深度可分离ConvGRU从当前浅层和上一状态共同生成，再在首次FST之前读回；状态可综合新特征，不被约束为warp/经验矩/RGB。独立空间控制每帧以当前嵌入代替历史，执行同一cell和readout；真实循环与同权重current干预均保留。它不是R64加通道/换offset，也不重新运行O1或任何旧STOP。

此基线的必要性是覆盖一个尚未完整检验的信息递推形式，而不是用普通ConvGRU组成论文创新。先只做固定CPU实现与学习通路资格，全部资格状态丢弃；正式预算、完整质量/独立历史/强post门和成本须另行登记。空间保留、同容量空间、真实历史贡献、强传统参考和后续独立外部/手机缺一不可。若新状态仍无跨组信号，停止其固定cell/宽度/dilation/初始化/目标/LR/长度/seed网格，不进入新的逐项救分。
