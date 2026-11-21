# L1/A1同输入证据归并

2026-11-21；Codex  (sqs_A100)。[结论与投资选择](findings.md)、[固定范围及敏感性补记](scope.md)、[0202](../../../docs/进展报告/0202-T-261121-同输入证据归并与时间信息预算问题.md)。本轮没有新像素、CNN、flow、优化或GPU任务，不改任何原门/STOP。

- [896完整配对/56统计](results/analysis001/tables.md)：共同32四帧训练抽样窗×两原F0 seed，原192基线行五指标跨研究差0。
- [全场景敏感性](results/scene-sensitivity001/tables.md)：13 SYN/9 ND场景；616场景项＋616留一项/56场景等权统计，无新独立测试/置信区间。历史增量弱的结论对逐一留出场景仍成立，部分ND质量/支配结论有权重范围。
- [近期直接先例81](../../5.prior-art-survey/docs/81-recent-offset-transport-and-reused-priors.md)：新增关键帧扩散偏移传输；CRISP/F2R/FilterFusion是已有阅读，不当新发现。
- [下一一帧等待问题](next-question.md)：仅问题和元数据源核查，尚未实现/登记正式科学协议或运行新指标。未来可用性/完整成本明确另列，不拿经典固定函数当原创方法。

论文目标active未完成，原论文/独立最终封存/所有旧STOP/7号/并行worktree保持。
