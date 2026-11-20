# T1：同F0起点的完整已知蒸馏对照

2026-11-20；Codex  (sqs_A100)。**已完整负结项，无活跃T1任务，不重启原链。** [决定与止损](decision.md)、[0180](../../../docs/进展报告/0180-T-26112019-同原骨干时序KD完整评价与止损.md)、[真实终态](process/terminal001.json)、[过程](process/journal.md)。论文goal仍active，普通KD不主张创新，V1及所有旧STOP保持。

先[核直接先例](prior-art.md)，再完成[训练目标预检](preflight-decision.md)四组必要前提，随后[另立完整协议](training-protocol.md)、[资格前冻42源](process/training-source-freeze001.json)。CPU6/CUDA6资格全丢弃，两个seed三学生gt/spatial/temporal均直接继承自身原4075参数F0完整权重，再各4000，共24000正式更新；六唯一最终有限重载，全训练/窗口/备份/终态保留，没有科学重试、追加预算或DEV选备份。

[全部240共同条件/48均值](report/joint/tables.md)：180新质量与180官方JOD各8064帧、复用原60参照。恢复保护3/4，主要A/B各0/40，完整A/B与成本扩展均False。时序教师相对空间KD在全部20条件降低时间误差，四组改善1.49%–2.38%，JOD均值小幅提高，但距F0两强post时间仍高约29%–48%，真实seed0恢复亦未保住。不用小幅正例覆盖主门槛，也不因负判定抹掉正例。

[19对/380条件/76均值完整差分](report/analysis/differences.md)、[四固定联系图与实际审阅](report/contacts/actual-review001.json)、[48点PNG/PDF/CSV与实际PNG审阅](report/plots/actual-review001.json)。图非播放/MOS，重复DEV不是独立最终，参数/解析MAC不当手机速度。

固定同F0/T4-DIS/等权KD的teacher、alpha、T、混合比例、loss、LR、长度/seed局部网格STOP，不扩失败成本/外部/手机。保留已知训练参照；新独立问题须先查重并说明结构/信息入口差异，再登记可证伪原型。所有原PID已实际退出、GPU释放，静态旧state不作重启依据。

实现、资格、首seed、全训练质量、全评分/止损各阶段分别密集提交和SSH推送。长等待确实按实测速率sleep，醒后核原PID/start/runtime，不设墙钟截止。全部资产位于zrh自有data，main/指定作者/no-ff、原84值/snapshot003/PDF020/独立最终封存、7号和其他worktree边界保持。
