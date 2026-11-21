# I1：Instant Video Models 的 MobileIE 任务适配

2026-11-21T15:43:51.687629+08:00；Codex  (sqs_A100)。research/O1。当前Simple learned双层级、双seed正式学习进行中，[协议](simple-protocol.md)、[执行002](simple-gpu-execution002.md)、[首次实际观察](process/simple-observation001.json)、[等待后观察](process/simple-observation002.json)、[0207](../../../docs/进展报告/0207-T-26112115-八帧稳定化基线资格与双种子正式学习.md)。这是已发表基线适配，不是本项目原创方法或作者HDRNet/NFS成绩复现。

原F0冻结，作者Simple模块按全部6个Conv2d＋root或仅root学习66/3衰减，总4141/4078参数；同B1T8/128、同批、原RMSE/λ.4、各80000×两seed共320000正式更新。新采样从原57SYN/12ND真实连续8帧，旧T4函数不改。CPU/CUDA资格各4丢弃更新已通过，资格权重/Adam全丢弃后fresh正式。

原父3965864/start118637026；数值seed0为3966953/start118647548/GPU6，seed1为3966961/start118647563/GPU7。16:02观察原身份活跃、最近完整13K/12K共同步，两个10K恢复备份均已核；按原runtime和实际/proc接续，不因旧日志/观察超时重启。1/2新鲜余量不足，所以首次GPU前修订至6/7并双重准入，不碰其它任务；无墙钟截止，资源暂停完整共同步恢复。同卡后来各新增约9.4GiB其它进程，按登记机制共享让步，独占速度不作当前完成估计。

解析Conv仍原7.937741G/1080p，但EMA/复制/状态不是零开销；FP32保留状态all约427MiB、output约24MiB，非峰值RAM/手机速度。正式数据目标和原作者不同，不能用更多训练步骤作为原创点或写同预算。资格和训练进度不是方法成功。

全长评价源码已准备/冻结；[CPU接续器](process/simple-after-launch001.json)3991883/start118916373只等原父正常完成，再评价/绘图各一次，待四唯一final后执行180序列/36均值/44门；没有新DEV结果/JOD/手机或独立最终。四图导出工具已准备，严格完整后生成，仍需逐张实际审阅；不用部分分数选参。任何层级失败按原协议停止对应局部网格，已发表基线结果仍保留。

此前完整controlled k1/k3的[CPU资格与费用决定](controlled-preflight-decision.md)保持：原生层控制器约193/258倍Conv、保留状态1.19GB，不在该高代价接入上正式训练/救分，也不说作者方法性能失败。[先例84](../../5.prior-art-survey/docs/84-instant-video-models-and-controlled-spatial-reference.md)、[完整过程](process/journal.md)。原作者23源/许可、所有旧STOP/原主表/最终/7号/并行worktree保持，论文目标active未完成。

接续更新：2026-11-21T16:07:22.448079+08:00；Codex  (sqs_A100)。原因：十分钟真实等待后补10K恢复备份及新同卡进程，未改科学预算或读取部分DEV。

接续更新：2026-11-21T16:22:18.524236+08:00；Codex  (sqs_A100)。原因：准备全量图表并接通训练正常终态后的CPU评价，无新模型/质量结果，不改原数值进程。

解释补充：2026-11-21T16:28:55.507116+08:00；Codex  (sqs_A100)。[完整层/输出归因边界](simple-interpretation001.md)：all的tail和root形成两级RGB EMA，output只有一级。因此原增量门通过仍不能单独证明内部特征历史贡献；保留原历史字段/44门，不新增实验网格。
