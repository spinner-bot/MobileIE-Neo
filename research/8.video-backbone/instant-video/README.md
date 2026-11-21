# I1：Instant Video Models 的 MobileIE 任务适配

2026-11-21T15:43:51.687629+08:00；Codex  (sqs_A100)。research/O1。当前Simple learned双层级、双seed正式学习进行中，[协议](simple-protocol.md)、[执行002](simple-gpu-execution002.md)、[首次实际观察](process/simple-observation001.json)、[0207](../../../docs/进展报告/0207-T-26112115-八帧稳定化基线资格与双种子正式学习.md)。这是已发表基线适配，不是本项目原创方法或作者HDRNet/NFS成绩复现。

原F0冻结，作者Simple模块按全部6个Conv2d＋root或仅root学习66/3衰减，总4141/4078参数；同B1T8/128、同批、原RMSE/λ.4、各80000×两seed共320000正式更新。新采样从原57SYN/12ND真实连续8帧，旧T4函数不改。CPU/CUDA资格各4丢弃更新已通过，资格权重/Adam全丢弃后fresh正式。

原父3965864/start118637026；数值seed0为3966953/start118647548/GPU6，seed1为3966961/start118647563/GPU7。最新观察同身份活跃、各4000共同步/8000更新；按原runtime和实际/proc接续，不因旧日志/观察超时重启。1/2新鲜余量不足，所以首次GPU前修订至6/7并双重准入，不碰其它任务；无墙钟截止，资源暂停完整共同步恢复。

解析Conv仍原7.937741G/1080p，但EMA/复制/状态不是零开销；FP32保留状态all约427MiB、output约24MiB，非峰值RAM/手机速度。正式数据目标和原作者不同，不能用更多训练步骤作为原创点或写同预算。资格和训练进度不是方法成功。

全长评价源码已准备/冻结，待四唯一final后执行180序列/36均值/44门；没有新DEV结果/JOD/手机或独立最终。固定四图待完整后生成实阅，不用部分分数选参。任何层级失败按原协议停止对应局部网格，已发表基线结果仍保留。

此前完整controlled k1/k3的[CPU资格与费用决定](controlled-preflight-decision.md)保持：原生层控制器约193/258倍Conv、保留状态1.19GB，不在该高代价接入上正式训练/救分，也不说作者方法性能失败。[先例84](../../5.prior-art-survey/docs/84-instant-video-models-and-controlled-spatial-reference.md)、[完整过程](process/journal.md)。原作者23源/许可、所有旧STOP/原主表/最终/7号/并行worktree保持，论文目标active未完成。
