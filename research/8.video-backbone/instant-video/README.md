# I1：Instant Video Models 的 MobileIE 任务适配

2026-11-21T15:08:21.775158+08:00；Codex  (sqs_A100)。research/O1。目的为补已发表因果稳定化学习基线，不是本项目原创方法。作者NeurIPS2025论文及固定代码的范围见[先例84](../../5.prior-art-survey/docs/84-instant-video-models-and-controlled-spatial-reference.md)、[读取记录](process/source-reading001.json)。[选定原源码与MIT许可](../../../third_party/instant-video-models/SOURCE.json)保持，非原HDRNet/NFS成绩复现。

[controlled接入资格/费用决定](controlled-preflight-decision.md)：完整Conv2d与输出的官方k1/k3都通过有限实现资格，原骨干冻结/反向路径正常。64个F0或被适配图、4丢弃更新，0新真实帧/flow/GPU；原3954845正常退出，无活跃任务。仅实现资格，尚无I1质量/时序或正式训练结果。

直接在MobileIE原生分辨率接入32/64控制器，1080p解析卷积1530.79/2046.77GMAC，为原F0的192.85/257.85倍；FP32保留状态解析约1.19GB，非峰值RAM/实测手机。决定不在这两个完整controlled接入上开启昂贵正式学习或缩宽/改尺度/删层救分，不说作者方法性能失败。

下一是同论文已发表Simple learned的全层与仅输出层级，先登记源/资格/训练预算；目前尚未实现或启动。没有新controller卷积仍不等于无状态/零推理开销，需实际学习和完整成本判断。[0206](../../../docs/进展报告/0206-T-261121-已发表稳定化基线接入与分辨率费用评估.md)、[完整过程](process/journal.md)。F1及全部旧STOP、原主表/最终/7号/并行worktree保持，论文目标active未完成。
