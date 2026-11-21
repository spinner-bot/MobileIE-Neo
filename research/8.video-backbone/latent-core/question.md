# L1：先检验可学习潜在状态，而非再调传输器

创建：2026-11-21T10:47:42.154772+08:00；作者：Codex  (sqs_A100)。分支research/O1。[O1完整决定](../operator-core/decision.md)及所有旧STOP保持，本文件不放宽其门槛或续训其模型。

|已有证据|已检验的状态/学习|尚不能推出|
|---|---|---|
|[M1](../multiaffine-core/spatial-decision.md)/V1|新随机/不同暖启动骨干恢复失守|所有视频骨干必然失去画质|
|[T1](../teacher-targets/decision.md)/[T2](../recurrent-objective/decision.md)|输出KD或固定DIS实际递归目标|所有状态机制已被检验|
|[R64](../../4.temporal-consistency/docs/exploration-r64-dense-history-training-protocol.md)|共同学习原dense、quarter8 project/warp凸融合|自由生成的循环特征无效|
|[R73](../../4.temporal-consistency/docs/exploration-r73-decision.md)|原head均值/二阶经验混合矩|一般隐状态必须表达为矩|
|[B1](../appearance-state/decision.md)|外观表索引历史|一般无显式位移的时间网络无效|
|[O1](../operator-core/decision.md)|共享描述/非局部点流/响应目标；恢复8/8，历史0/8|confidence全关、位移不准或唯一根因|

**下一唯一问题**：原F0暖起点上，一个不被点流/凸样本混合/经验矩约束的低分辨率学习状态，能否保住恢复并产生超出同容量空间及同权重current的历史收益，最终接近强FB/DIS而减少完整成本？新状态允许自身卷积生成和负特征；不预言运动已解决或任意长时间稳定。

原始4075不是硬上限。L1固定quarter32、深度可分离ConvGRU与原head内读回，参数预计12431、1080p解析Conv8.982835344G、FP32持久state16588800B。多于原模型的权重/计算明确披露；归一化、sigmoid/tanh、pool/resize/copy不在Conv量内，实际峰值RAM/整链CPU/手机仍未知。不能用无flow代替速度证据。

[先例79](../../5.prior-art-survey/docs/79-latent-state-recurrence-and-transport-boundaries.md)已排除一般RNN/GRU/隐状态/乘法为新颖性；L1是有限的已知机制任务基线，不是当前论文方法。CPU资格通过也只说明路径可运行，随后另立正式协议；失败保留，不调宽度/dilation/门/初始化或换loss救资格。研究需继续形成真正可验证贡献，原84值/PDF020/最终/7号/并行main及其他worktree不动。
