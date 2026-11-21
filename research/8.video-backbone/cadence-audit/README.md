# 来源索引与时间基准核查

2026-11-21；Codex  (sqs_A100)。已完成；[结论](findings.md)、[全部76来源表](results/audit001/source-index-summary.csv)、[时间契约](results/audit001/time-contract.json)、[0204](../../../docs/进展报告/0204-T-261121-来源帧索引与时间基准核查.md)。0新像素/视频解码/CNN/flow/训练/GPU，不重评旧指标。

本地OLiVES/DAVIS都保留连续文件前缀，没有源码层面的时间抽样/重编号；L1固定32个四帧draw也连续。OLiVES作者报告25fps；DAVIS2016报告24fps，只能适用于官方清单确认的源，不能推广到2017新增源。D1 bear/lucia属于2016，walking不属于；未知源保持登记帧单位。三手机既有解码位置中两源有变间隔，不把它们等同统一sensor时钟或手机推理延迟。

[原范围](scope.md)、[有界官方split文本读取](process/davis-split-read001.json)、[主来源读取](process/source-reading001.json)及完整结果分开归档，源端导出/实际PTS未独立验证的边界明确。旧D1/所有STOP、名义30fps JOD/原论文/最终封存不改；下一回到已知固定时滞递归平滑先例和信息/成本前提，尚无新实验或创新方法。论文目标active未完成。
