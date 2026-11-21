# I1：Instant Video Models 的 MobileIE 任务适配

2026-11-21T20:25:17.183021+08:00；Codex  (sqs_A100)。触发用户阶段报告请求；research/O1。[最新0208](../../../docs/进展报告/0208-T-26112120-阶段研究进展与当前目标.md)、[完整决定](simple-decision001.md)、[真实终态](process/simple-terminal001.json)。此前启动的Simple learned双seed训练、自动CPU完整评价及绘图均已自然完成，原句柄已终态；本次不启动新实验。用户已清除自主目标，以下旧报告的active或正在训练是历史状态，不能据此恢复工作。

原F0冻结，作者Simple模块按全部6个Conv2d＋root(all)或仅root(output)学习66/3衰减，总4141/4078参数。[事前协议](simple-protocol.md)固定真实连续B1T8/128、同批、原RMSE/λ.4，两层级各80K×两seed，全部320000正式更新/2560000学生帧/四唯一final齐备。原57SYN/12ND源，CPU/CUDA资格各4丢弃更新不继承。GPU1/2余量不足，首次CUDA前[执行002](simple-gpu-execution002.md)改同组6/7，仍双重准入/按共享让步，无墙钟截断。数值seed1在18:13:54、seed0在18:28:36正常完成；不按旧PID重启。

[完整结果](results/simple-eval001/tables.md)：180序列/8064逐帧/36均值/44比较，原十个重复DEV条件/两seed/九输出。all/output对F0组均值时间改善约12.38%–14.56%/10.20%–12.27%，但质量分别低.776–1.351/.938–1.367dB，原强post仍有明显优势。两个层级路线各0/20，完整堆叠增量0/4，全部0/44；局部正例完整保留，不将零联合通过写成所有单项无收益。

按原协议停止这两个固定层级/τ8/λ.4/init4/当前目标与LR/80K及补救网格，不扩大失败成本/JOD/外部/手机。当前没有可进入下一费用与感知评价的层级。[DEV前归因边界](simple-interpretation001.md)保持：all含tail/root两级RGB EMA，output只有一级，差异不单独证明内部特征历史作用。普通EMA不是原创，也不是原作者HDRNet/NFS结果复现；本任务负结果不否定整个IVM。

[四图及全部220配对](report/simple-analysis001/)已导出，[四PNG实际审阅](report/simple-analysis001/actual-review001.json)另记录，PDF只导出。静态联系图不是视频感知证据；重复DEV仅三SYN来源/四ND来源，非独立最终。完整GPU训练日志和元数据已归档，大权重/数组仍仅在zrh自有data，数值源码不改。

解析Conv仍原7.937741G/1080p，但EMA/复制/状态不是零开销；FP32保留状态all约427MiB、output约24MiB，非手机/峰值RAM。[此前controlled决定](controlled-preflight-decision.md)保持：原生接入约193/258倍Conv、1.19GB保留状态，未正式训练/救分，也不称作者质量失败。[先例84](../../5.prior-art-survey/docs/84-instant-video-models-and-controlled-spatial-reference.md)、[全过程](process/journal.md)。所有旧STOP/原84质量值/PDF020/独立最终/7号与并行worktree保持；论文目标未实现，下一原创候选尚未登记。
