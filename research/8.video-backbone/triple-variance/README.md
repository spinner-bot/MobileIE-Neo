# N1：增强后随机波动能否由三观测识别

2026-11-20；Codex  (sqs_A100)。承接S1/P1负结项，先验证统计来源，不直接训练新滤波器。three-cornered-hat与时间噪声估计已有先例，不主张新统计理论。

[先例/恒等式/适用边界](../../5.prior-art-survey/docs/75-triple-difference-noise-identification-boundaries.md)、[预登记](protocol.md)、[统计模块](triple_statistics.py)、[唯一CPU运行](probe.py)、[16具名源/输入冻结](process/source-freeze001.json)。原F0双seed、四既定训练纹理，2048独立参考图+9216三观测图=11264NN；静止、0优化/flow/GPU/DEV/最终，不与视频质量或手机速度混称。

必须同时验证独立条件下所有24组的全局误差、局部单triplet误差及负估计比例；包含异photon导致的增强器条件均值变化。另24相关观测组完整保留，只作假设违背诊断，不错误要求它们估计总方差。通过也仅允许另登记有限实际运动校准；失败不调tile/窗口/截断或训练补救。

原去偏、经验矩、RMS门、单向传输和全部旧STOP保持，原论文84值/PDF020、最终封存、7号和其他worktree不动。资产经自有.cache，代码/资料/结果/过程/报告单独密集提交和阶段推送，长等待按实测sleep。
