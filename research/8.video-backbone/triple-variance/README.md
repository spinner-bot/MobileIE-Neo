# N1：增强后随机波动能否由三观测识别

2026-11-20；Codex  (sqs_A100)。承接S1/P1负结项，先验证统计来源，不直接训练新滤波器。three-cornered-hat与时间噪声估计已有先例，不主张新统计理论。

已完整结项：[决定](decision.md)、[0187](../../../docs/进展报告/0187-T-26112023-三观测方差识别校准与骨干问题收敛.md)、[原终态](process/terminal001.json)、[过程](process/journal.md)。全11264NN正常111.089698秒、0更新/GPU/DEV/最终，主24组完整门12/24，实际运动资格False。固定photon8/8通过保留，异photon只有4/16；均值与协方差污染不能忽略。停止固定三观测/16tile/截断/窗口/模式/相关构造网格，不训练补救；两个全点PNG实际查看，PDF仅导出。全部原任务终态。[下一骨干问题](next-core-question.md)只是待核多线性来源树，未实现/登记或宣称创新。

[先例/恒等式/适用边界](../../5.prior-art-survey/docs/75-triple-difference-noise-identification-boundaries.md)、[预登记](protocol.md)、[统计模块](triple_statistics.py)、[唯一CPU运行](probe.py)、[16具名源/输入冻结](process/source-freeze001.json)。原F0双seed、四既定训练纹理，2048独立参考图+9216三观测图=11264NN；静止、0优化/flow/GPU/DEV/最终，不与视频质量或手机速度混称。

必须同时验证独立条件下所有24组的全局误差、局部单triplet误差及负估计比例；包含异photon导致的增强器条件均值变化。另24相关观测组完整保留，只作假设违背诊断，不错误要求它们估计总方差。通过也仅允许另登记有限实际运动校准；失败不调tile/窗口/截断或训练补救。

原去偏、经验矩、RMS门、单向传输和全部旧STOP保持，原论文84值/PDF020、最终封存、7号和其他worktree不动。资产经自有.cache，代码/资料/结果/过程/报告单独密集提交和阶段推送，长等待按实测sleep。
