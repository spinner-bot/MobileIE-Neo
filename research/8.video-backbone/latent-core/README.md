# L1：保留原恢复起点的可学习潜在状态基线

创建：2026-11-21T10:55:50.388409+08:00；Codex  (sqs_A100)。分支research/O1。[问题与证据矩阵](question.md)、[直接先例79](../../5.prior-art-survey/docs/79-latent-state-recurrence-and-transport-boundaries.md)、[CPU资格](preflight-decision.md)、[正式预登记](training-protocol.md)。

L1把32通道quarter状态作为生成的循环潜在特征，标准分离ConvGRU从当前浅层及旧state更新，在首次FST前读回；完整原F0权重起点保留。独立空间每帧用当前嵌入替历史，同cell/readout。普通RNN/GRU/隐状态/函数保持有直接先例，这是待质量验证的已知机制任务基线，不是新方法。

CPU原3836298自然29.622781秒/exit0/已无。四初始F0差0，循环末帧过去梯度7.741090/4.080715、空间0；stream差≤1.192093e−7，所有第二步共享及新路径梯度有限非零。26具名资格源保持，32F0/196学生图、8更新全丢弃，无导出/正式/GPU/DEV。资格不能证明质量/闪烁/手机或创新。

12431参数，解析1080p Conv8.982835344G，FP32持久state16588800B，128²131072B；不是完整MAC/峰值RAM/手机时延或零开销。原O1固定配置STOP及旧STOP不改，未续训其权重或参数网格。

当前更新：2026-11-21T11:28:25.154314+08:00；Codex  (sqs_A100)。[0199](../../../docs/进展报告/0199-T-261121-潜在状态CUDA资格与双种子正式学习启动.md)：原CUDA007正常4更新全丢弃/不导出，实际GPU1/allocator约286.6MB；原外层33516 exit143/父无且无runtime单独保留。新增1执行恢复源，从已过资格边界以独立session新父3852634/start117153546调用原seed driver，原26/44源全不改、不重复资格。fresh seed0 worker3852927/start117158126在GPU1、seed1 worker3852924/start117158111在GPU2，初始计数均0，[观察002](process/observation002.json)核实际live/UUID及各209/205共同步。固定8K×4/32K、512K学生/0formal教师及完整同32窗/11输出/self4/44比较保持，尚无完整质量结果。后续核同原PID/start/runtime，长sleep按实测速率估，不按旧stage重启；原chain001已终态不再接续。

接续[过程](process/journal.md)及[0197](../../../docs/进展报告/0197-T-261121-潜在状态基线资格与固定学习方案.md)。下一完成原预算的正式链并取得真实质量证据，不再扩大普通组件诊断；若失败按协议停止局部网格。原84/PDF020/封存最终/7号/并行main及其他worktree保持，论文目标active未完成，密集提交/阶段推送/未来--no-ff。
