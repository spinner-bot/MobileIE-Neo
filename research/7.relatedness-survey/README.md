# MobileIE 相关性调研工作站

创建：2026-11-17T18:05:52+08:00；作者：Codex  (sqs_A100) <langxibielangle@qq.com>；触发：用户确认调研任务、收口口径及证据受限结项。


分支 `survey/pre`；源基点 `demo/org@e3803178337f5951a0e1138587317ed816e64d01`。

当前状态：Gate1已于2026-11-20按冻结标准完成审查，收集快照截至2026-11-17，共113项候选；[正式审查](docs/0003-Gate1审查-26112012.md)。尚未完成逐项Gate2与总体结论报告。

2026-11-20阶段进度：113项永久索引和初级排序已冻结；MIE-0001、0002、0003已顺序通过Gate2（公开文献／适用核心静态分析，非数值复现），正在分析MIE-0004。最新状态以[顺序日志](catalog/analysis-log.jsonl)和各项Gate2原记录为准。本段是阶段快照，不代表其后状态自动回写。

当前增补状态：发现旧截止日前新增引用及AnyMod正式版本核查缺口，已启用[G1-A0001增补暂停推进记录](collection/gate1-addenda/0001/status.json)。原113项索引/冻结记录保留；11项新身份尚未排序编号，MIE-0004暂缓结项，先补收集及重新审查。实际状态以增补记录和顺序日志为准。

任务：为 MobileIE 照片→视频和 ICCV 2027 科研提供全面、可追溯的相关性证据。现有5号调研仅为历史线索，不继承其 Gate 通过状态或“没有相关工作”的结论。

- [任务边界](docs/0001-任务与决策-26111718.md)
- [Gate1标准](gates/gate1-v1.md)：冻结后才启动本轮外部收集。
- [Gate2标准](gates/gate2-v1.md)：冻结后逐项执行；当前项未结项不进入下一项。
- [进展与决策](docs/0002-T-26111718-推进记录.md)
- [永久资源索引与阅读优先级](catalog/README.md)
- 已结项：[MobileIE原论文](resources/MIE-0001/derived/report/analysis.md)、[官方实现](resources/MIE-0002/derived/report/analysis.md)、[AnyMod-LLVE论文](resources/MIE-0003/derived/report/analysis.md)。缺口保留在各项，不把Gate通过当论文／设备结果验证成功。
- `collection/`：来源覆盖矩阵、检索原始回执、去重／排除／追踪台账与 Gate1 审查。
- `resources/`：Gate1后分配静态ID；每项含原始资源入口、衍生过程、结论及独立Gate2记录。
- `report/`：全部资源结项后生成综合Markdown与HTML报告。
- `scripts/`：可复核的获取、核验、建表工具。只读收集，不执行外部研究代码。

原始大资料按现有规范归档在zrh自有 `data/MobileIE-Neo/cache/relatedness-survey/`，仓库保存可解析入口、源URL、时间、摘要与获取结果；检索回执、元数据、过程、结论及轻量证据入Git。不得提交凭据，不修改既有specs。不在本阶段训练或执行Gate3。
