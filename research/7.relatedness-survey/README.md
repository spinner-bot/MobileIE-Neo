# MobileIE 相关性调研工作站

创建：2026-11-17T18:05:52+08:00；作者：Codex  (sqs_A100) <langxibielangle@qq.com>；触发：用户确认调研任务、收口口径及证据受限结项。


分支 `survey/pre`；源基点 `demo/org@e3803178337f5951a0e1138587317ed816e64d01`。

任务：为 MobileIE 照片→视频和 ICCV 2027 科研提供全面、可追溯的相关性证据。现有5号调研仅为历史线索，不继承其 Gate 通过状态或“没有相关工作”的结论。

- [任务边界](docs/0001-任务与决策-26111718.md)
- [Gate1标准](gates/gate1-v1.md)：冻结后才启动本轮外部收集。
- [Gate2标准](gates/gate2-v1.md)：冻结后逐项执行；当前项未结项不进入下一项。
- [进展与决策](docs/0002-T-26111718-推进记录.md)
- `collection/`：来源覆盖矩阵、检索原始回执、去重／排除／追踪台账与 Gate1 审查。
- `resources/`：Gate1后分配静态ID；每项含原始资源入口、衍生过程、结论及独立Gate2记录。
- `report/`：全部资源结项后生成综合Markdown与HTML报告。
- `scripts/`：可复核的获取、核验、建表工具。只读收集，不执行外部研究代码。

原始大资料按现有规范归档在zrh自有 `data/MobileIE-Neo/cache/relatedness-survey/`，仓库保存可解析入口、源URL、时间、摘要与获取结果；检索回执、元数据、过程、结论及轻量证据入Git。不得提交凭据，不修改既有specs。不在本阶段训练或执行Gate3。
