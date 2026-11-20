# 主干信息流与时域化

2026-11-20；Codex  (sqs_A100)。main主题8，research/7继续留给其他分支。

V1已完整结项为负结果：[决定与止损](decision-v1.md)、[0175报告](../../docs/进展报告/0175-T-26112017-完整视频骨干候选评价与止损.md)、[原终态](process/v1/terminal001.json)。两seed全28000正式训练/八唯一最终、220质量/220官方JOD各9856逐帧、280共同条件/56均值及四固定图均正常，四联系图和两全表图已实际查看。全部旧worker/coordinator均退出，无活跃V1任务，不重启旧链。

空间保护0/4，主要A/B各0/28、完整资格均False。独立空间相对F0低2.29–3.98dB；学习和固定因果输出在全部20条件都被其自身DIS post三指标支配。相对自身preview仍有19/20时间改善等局部正例，详[全部成对结果](report/v1-analysis/tables.md)，不否定所有历史信息或低计算骨干。按原协议停止固定核/门的宽度、phase倍率、alpha、噪声图、loss、LR、长度/seed网格，不扩该失败候选成本/外部/手机。

[完整共同表](report/v1-joint/tables.md)、[全点CSV与PDF/PNG](report/v1-plots/)、[联系图](report/v1-contacts/)、[实际图表审阅](report/v1-plots/actual-review001.json)均保留；重复DEV/合成来源/2seed不构成独立视频或总体显著性证据。静态图不是全视频/MOS，也不证明混叠原因。没有可投稿方法、手机速度或独立最终结论，整篇论文目标仍active。

后续独立研究须先保住原有增强能力，明确表示/初始化转移/训练预算与直接先例，再检验时域机制。原F0继承官方LoL→R41 4K→R44 4K；新学生2K KD＋4K GT不等于同样预训练历史。当前负结果不能单独归因为packing/蒸馏/欠训练，也不能用延长同一学生或改门/phase/loss来救分。原MobileIE stride1无下采样，普通packing/KD/门/递归已有先例，不构成已确立创新。

复核来源：[原模型与训练审计](original-model-and-training.md)、[先例与原候选](precedents-and-candidate.md)、[原协议](protocol-v1.md)、[CPU执行修订](cpu-backend-amendment001.md)、[GPU组执行修订002](gpu-group-amendment002.md)、[完整过程](process/v1/journal.md)。空间16180/学习16365参数，较低解析MAC不是速度，不能叫4K或零推理开销。全部原34＋5＋CPU4/执行001的2源/002的4源保持，不改写历史失败。

按用户要求，长等待依据实际初段速度估算sleep，覆盖大部分预计余量，醒后核同PID/start/runtime与自动接续；到期不作重启或算法截止。main/自有data/指定作者/no-ff/小步提交和阶段推送，所有旧STOP/独立最终封存及survey/pre/demo/org边界保持。
