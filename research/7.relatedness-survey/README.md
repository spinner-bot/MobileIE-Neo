# MobileIE 相关性调研工作站

创建：2026-11-17T18:05:52+08:00；作者：Codex  (sqs_A100) <langxibielangle@qq.com>；触发：用户确认调研任务、收口口径及证据受限结项。

阶段工作汇报（完成于2026-11-21）：[HTML完整版](report/0001-阶段工作汇报-20261120.html)｜[Markdown原文](report/0001-阶段工作汇报-20261120.md)｜[验证记录](report/0001-visual-review.md)。本报告快照为11月20日，完整列出124项与13项结项、当前缺口和待办；该报告保留暂停时快照。用户于2026-11-21明确重新设定目标并恢复调研；G1-A0002已完成审查并发布v3队列，旧报告不回写。


分支 `survey/pre`；源基点 `demo/org@e3803178337f5951a0e1138587317ed816e64d01`。

当前状态：原113项、G1-A0001新增11项及[G1-A0002新增PASTA论文](docs/0007-Gate1二次增补审查-26112107.md)均按冻结Gate1审查，共125个身份（51论文、65代码、9其他）。截止快照仍2026-11-17，实际补取含11月20/21日。原Gate、历史目录和结项记录保留，尚未完成全部Gate2和总体结论报告。

当前分析进度：13/125项已顺序完成Gate2；MIE-0003另已完成[正式ICML/PMLR追加复审](resources/MIE-0003/derived/reviews/0001/report/analysis.md)。MIE-0014 DRNet为唯一已开始未结项资源，Gate1增补推进限制已解除，恢复其来源核查与Gate2。下一项仍MIE-0015；新MIE-0125的分析序为23，不提前深入分析。以[目录v3](catalog/README-v3.md)、[顺序日志](catalog/analysis-log.jsonl)及[追加复审日志](catalog/review-log.jsonl)为准。原1–124静态ID、v1/v2目录与旧审查不变。

这些Gate2结项属于公开文献与适用核心静态分析，不是数值复现。总体报告与Gate3预实验尚未完成，本阶段不执行Gate3。

任务：为 MobileIE 照片→视频和 ICCV 2027 科研提供全面、可追溯的相关性证据。现有5号调研仅为历史线索，不继承其 Gate 通过状态或“没有相关工作”的结论。

- [任务边界](docs/0001-任务与决策-26111718.md)
- [Gate1标准](gates/gate1-v1.md)：冻结后才启动本轮外部收集。
- [Gate2标准](gates/gate2-v1.md)：冻结后逐项执行；当前项未结项不进入下一项。
- [进展与决策](docs/0002-T-26111718-推进记录.md)
- [永久资源索引与阅读优先级](catalog/README-v3.md)
- 已结项：[MobileIE原论文](resources/MIE-0001/derived/report/analysis.md)、[官方实现](resources/MIE-0002/derived/report/analysis.md)、[AnyMod-LLVE原v1分析](resources/MIE-0003/derived/report/analysis.md)、[正式版复审](resources/MIE-0003/derived/reviews/0001/report/analysis.md)、[AMNet官方项目](resources/MIE-0004/derived/report/analysis.md)、[Ascend NPU接续](resources/MIE-0005/derived/report/analysis.md)、[Rust工程](resources/MIE-0006/derived/report/analysis.md)、[Android编码变体](resources/MIE-0007/derived/report/analysis.md)、[Android BYOM相机工程](resources/MIE-0008/derived/report/analysis.md)、[reef-track视频应用](resources/MIE-0009/derived/report/analysis.md)、[实时水下增强论文](resources/MIE-0010/derived/report/analysis.md)、[作者MRDConv发布模块](resources/MIE-0011/derived/report/analysis.md)、[FGDPA频率引导论文](resources/MIE-0012/derived/report/analysis.md)、[FGDPA作者实现](resources/MIE-0013/derived/report/analysis.md)。缺口保留在各项，不把Gate通过当论文／设备结果验证成功。
- `collection/`：来源覆盖矩阵、检索原始回执、去重／排除／追踪台账与 Gate1 审查。
- `resources/`：Gate1后分配静态ID；每项含原始资源入口、衍生过程、结论及独立Gate2记录。
- `report/`：全部资源结项后生成综合Markdown与HTML报告。
- `scripts/`：可复核的获取、核验、建表工具。只读收集，不执行外部研究代码。

原始大资料按现有规范归档在zrh自有 `data/MobileIE-Neo/cache/relatedness-survey/`，仓库保存可解析入口、源URL、时间、摘要与获取结果；检索回执、元数据、过程、结论及轻量证据入Git。不得提交凭据，不修改既有specs。不在本阶段训练或执行Gate3。
