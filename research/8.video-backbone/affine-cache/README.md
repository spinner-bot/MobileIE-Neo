# A1：关键帧局部仿射算子缓存

2026-11-21；Codex  (sqs_A100)。在research/O1完成一次固定零训练筛查，**完整未通过并停止**：[决定](decision.md)、[协议](protocol.md)、[全48均值与20门](results/screen001/tables.md)、[实际审阅PNG/PDF与512配对](report/analysis001/)。

原F0输出拟合quarter局部RGB仿射场，逐帧作用当前低光；比较每帧重拟合、隔帧更新系数、隔帧保持图像，各配独立FB/DIS强post。三类既有技术不作为创新。全部32唯一训练场景四帧抽样窗×两原F0 seed，实际新CNN/优化/GPU均0；逻辑省一半F0调用不代表实测整链/手机提速。

表示保护0/4、强组合0/16，SYN即使每帧拟合仍低约2.73–2.81dB，ND时间下降也不满足质量保护。固定表示/拟合/周期/补救网格STOP，不扩失败成本/训练/DEV/JOD/手机。54.77秒正常结束，原进程已无；[21具名冻结](process/source-freeze001.json)/[原终态](process/terminal001.json)/[过程](process/journal.md)保留。论文目标active未完成，原论文和封存最终不改。
