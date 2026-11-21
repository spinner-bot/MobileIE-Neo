# MobileIE 相关性调研工作站

创建：2026-11-17T18:05:52+08:00；作者：Codex  (sqs_A100) <langxibielangle@qq.com>；触发：用户确认调研任务、收口口径及证据受限结项。

阶段工作汇报（完成于2026-11-21）：[HTML完整版](report/0001-阶段工作汇报-20261120.html)｜[Markdown原文](report/0001-阶段工作汇报-20261120.md)｜[验证记录](report/0001-visual-review.md)。本报告快照为11月20日，完整列出124项与13项结项、当前缺口和待办；该报告保留暂停时快照。用户于2026-11-21明确重新设定目标并恢复调研；G1-A0002已完成审查并发布v3队列，旧报告不回写。


分支 `survey/pre`；源基点 `demo/org@e3803178337f5951a0e1138587317ed816e64d01`。

当前状态：原113项、G1-A0001新增11项及[G1-A0002新增PASTA论文](docs/0007-Gate1二次增补审查-26112107.md)均按冻结Gate1审查，共125个身份（51论文、65代码、9其他）。截止快照仍2026-11-17，实际补取含11月20/21日。原Gate、历史目录和结项记录保留，尚未完成全部Gate2和总体结论报告。

当前分析进度：30/125项已顺序结项（20项pass_full、10项closed_limited）；MIE-0003另已完成[正式ICML/PMLR追加复审](resources/MIE-0003/derived/reviews/0001/report/analysis.md)。MIE-0014 DRNet已完成[证据受限结项](resources/MIE-0014/derived/report/analysis.md)：完整预印本与适用实现已分析、两份权重完整恢复，正式版/盲配置/历史任务映射缺口保留。MIE-0015作者项目已完成[公开适用静态审查](resources/MIE-0015/derived/report/analysis.md)，MIE-0016 YUV低光论文已完成[证据受限结项](resources/MIE-0016/derived/report/analysis.md)：完整9页原文和发布工件已分析，可训练作者源/配置与全表模型对应未提供。MIE-0017发布项目已完成[证据受限结项](resources/MIE-0017/derived/report/analysis.md)，同家族原source/config缺口继续保留。MIE-0018 LiteIE论文已完成[证据受限结项](resources/MIE-0018/derived/report/analysis.md)：三个预印本版本与当前全部作者工件已核，正式全文/原训练IRM/手机链缺口保留。MIE-0019官方项目已完成[完整当前公开载体静态分析](resources/MIE-0019/derived/report/analysis.md)，不解除18原论文缺口。MIE-0020 MobileIE-6Ch已完成[当前发布载体静态分析](resources/MIE-0020/derived/report/analysis.md)，参数/历史提交/验证协议及手机视频缺口留档。MIE-0059 ULightIF已完成[证据受限结项](resources/MIE-0059/derived/report/analysis.md)：正文/原协议/图表未得，作者全部当前工件及获取边界已分析。MIE-0114作者项目已完成[当前公开载体静态分析](resources/MIE-0114/derived/report/analysis.md)，不解除59正式/历史缺口。MIE-0125 PASTA已完成[证据受限结项](resources/MIE-0125/derived/report/analysis.md)：19页主体全文与6图6表已读，独立补充/实现/权重及完整协议缺口保留。MIE-0021 FastSHADE已完成[正式公开论文分析](resources/MIE-0021/derived/report/analysis.md)：直接继承MBRConv/FP64折叠、单帧去噪与手机成本口径及图表内部问题留档。MIE-0022 JND轻量预滤波已完成[证据受限结项](resources/MIE-0022/derived/report/analysis.md)：主体五页及适用实现已分析，作者逐视频CSV实际内容和历史版本/成本证据缺口保留。MIE-0023 FJNDF作者项目已完成[证据受限结项](resources/MIE-0023/derived/report/analysis.md)：当前Git载体和适用源码已分析，平台外补充结果仍缺内容。MIE-0024 QATIE已完成[充分公开文献/静态载体分析](resources/MIE-0024/derived/report/analysis.md)：训练部署差异、量化剩余损失、实际工件布局与协议边界已留档。MIE-0025 QATIE作者项目已完成[充分公开项目静态分析](resources/MIE-0025/derived/report/analysis.md)：同源SHA阅读复用、命令入口与dtype/layout/后端及指标范围已核。MIE-0026 REZS-Edge已完成[证据受限结项](resources/MIE-0026/derived/report/analysis.md)：官方摘要/出版方引用DOI已核，正文/图表/实现及具体协议未得。MIE-0027 CBGAN已完成[证据受限结项](resources/MIE-0027/derived/report/analysis.md)：开放许可/直接引用已核，正文/图表/实现、具体变换与时序/成本协议未得。当前唯一活跃是分析序31的MIE-0028；永久ID与阅读序分开，按v3队列推进。以[目录v3](catalog/README-v3.md)、[顺序日志](catalog/analysis-log.jsonl)及[追加复审日志](catalog/review-log.jsonl)为准。原1–124静态ID、v1/v2目录与旧审查不变。

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
