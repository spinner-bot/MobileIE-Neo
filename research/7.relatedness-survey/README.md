# MobileIE 相关性调研工作站

创建：2026-11-17T18:05:52+08:00；作者：Codex  (sqs_A100) <langxibielangle@qq.com>；触发：用户确认调研任务、收口口径及证据受限结项。

阶段工作汇报（完成于2026-11-21）：[HTML完整版](report/0001-阶段工作汇报-20261120.html)｜[Markdown原文](report/0001-阶段工作汇报-20261120.md)｜[验证记录](report/0001-visual-review.md)。本报告快照为11月20日，完整列出124项与13项结项、当前缺口和待办；用户已暂停并清除原调研目标，本次仅交付阶段汇报，未自动恢复调研或解除G1-A0002。


分支 `survey/pre`；源基点 `demo/org@e3803178337f5951a0e1138587317ed816e64d01`。

当前状态：原113项与G1-A0001新增11项均已按冻结Gate1标准审查，共124个身份（50论文、65代码、9其他），截止快照为2026-11-17，实际补取日期为11月20日。见[原审查](docs/0003-Gate1审查-26112012.md)与[增补审查](docs/0005-Gate1增补审查-26112016.md)。尚未完成全部逐项Gate2与总体结论报告。

当前分析进度：13/124项已顺序完成Gate2；MIE-0003另已完成[正式ICML/PMLR版本追加复审](resources/MIE-0003/derived/reviews/0001/report/analysis.md)。[G1-A0001](collection/gate1-addenda/0001/status.json)补充收集和必需复审均已关闭，MIE-0004…0009已完成核心静态审查，MIE-0010实时水下增强论文已完成全文/图表与作者示例分析，MIE-0011作者项目已完成公开发布模块审阅，MIE-0012频率引导双路径注意力论文已完成全文/图表/适用作者实现审查，MIE-0013作者实现已完成公开适用源码审查，MIE-0014 DRNet动态重参数化论文已按顺序启动。新增MIE-0114…0124已编目，原113个ID及旧审查不变，当前以[目录v2](catalog/README-v2.md)、[顺序日志](catalog/analysis-log.jsonl)及[追加复审日志](catalog/review-log.jsonl)为准。

当前队列暂受[G1-A0002线索处置核查](collection/gate1-addenda/0002/status.json)约束：MIE-0014保持唯一活跃，先补核PASTA具体关系及必要Gate1增补审查，不进入下一资源。

这些Gate2结项属于公开文献与适用核心静态分析，不是数值复现。总体报告与Gate3预实验尚未完成，本阶段不执行Gate3。

任务：为 MobileIE 照片→视频和 ICCV 2027 科研提供全面、可追溯的相关性证据。现有5号调研仅为历史线索，不继承其 Gate 通过状态或“没有相关工作”的结论。

- [任务边界](docs/0001-任务与决策-26111718.md)
- [Gate1标准](gates/gate1-v1.md)：冻结后才启动本轮外部收集。
- [Gate2标准](gates/gate2-v1.md)：冻结后逐项执行；当前项未结项不进入下一项。
- [进展与决策](docs/0002-T-26111718-推进记录.md)
- [永久资源索引与阅读优先级](catalog/README-v2.md)
- 已结项：[MobileIE原论文](resources/MIE-0001/derived/report/analysis.md)、[官方实现](resources/MIE-0002/derived/report/analysis.md)、[AnyMod-LLVE原v1分析](resources/MIE-0003/derived/report/analysis.md)、[正式版复审](resources/MIE-0003/derived/reviews/0001/report/analysis.md)、[AMNet官方项目](resources/MIE-0004/derived/report/analysis.md)、[Ascend NPU接续](resources/MIE-0005/derived/report/analysis.md)、[Rust工程](resources/MIE-0006/derived/report/analysis.md)、[Android编码变体](resources/MIE-0007/derived/report/analysis.md)、[Android BYOM相机工程](resources/MIE-0008/derived/report/analysis.md)、[reef-track视频应用](resources/MIE-0009/derived/report/analysis.md)、[实时水下增强论文](resources/MIE-0010/derived/report/analysis.md)、[作者MRDConv发布模块](resources/MIE-0011/derived/report/analysis.md)、[FGDPA频率引导论文](resources/MIE-0012/derived/report/analysis.md)、[FGDPA作者实现](resources/MIE-0013/derived/report/analysis.md)。缺口保留在各项，不把Gate通过当论文／设备结果验证成功。
- `collection/`：来源覆盖矩阵、检索原始回执、去重／排除／追踪台账与 Gate1 审查。
- `resources/`：Gate1后分配静态ID；每项含原始资源入口、衍生过程、结论及独立Gate2记录。
- `report/`：全部资源结项后生成综合Markdown与HTML报告。
- `scripts/`：可复核的获取、核验、建表工具。只读收集，不执行外部研究代码。

原始大资料按现有规范归档在zrh自有 `data/MobileIE-Neo/cache/relatedness-survey/`，仓库保存可解析入口、源URL、时间、摘要与获取结果；检索回执、元数据、过程、结论及轻量证据入Git。不得提交凭据，不修改既有specs。不在本阶段训练或执行Gate3。
