# R28监督来源勘误：对应教师不等于输出蒸馏

2026-11-17；Codex  (sqs_A100)。触发：R82收口后核查新学习问题，发现R80协议把R28称作“固定学生后处理蒸馏”，该表述不准确。此为独立勘误，不修改已冻结协议、代码、权重、原结果或它们的摘要。

[R28实际训练](../scripts/train_transport_fst_r28.py)118行用冻结teacher和低光对准备对应场，139–146行则计算prediction与原target的Charbonnier/MSE。target在此前由real_pair/synthetic_pair构造的GT组成；没有把强rec输出当作训练像素目标。R28的current映射也是同权重替代部署控制，不是由单独后处理输出蒸馏而得的学生。原协议本身写明GT重建、无辅助项，原训练及失败决定不因此改变。

R80协议开头的“R28已有固定学生后处理蒸馏”不能作为已经做过强递归输出KD的证据。note64所说的R28“全局固定学生权重”只宜理解为普通训练后的全局固定权重，与逐视频前缀系数状态比较，不应据此补出蒸馏实验。R80真实前缀伪目标/ridge实现和完整失败结果保持。

这项更正也不使普通KD成为创新：2020 Optical Flow Distillation §3.2已经给出有flow/无flow教师输出差与学生相对基线变化的匹配；近期PocketDVDNet也使用teacher/GT重建混合。若教师与学生的减数都是同一固定F0，`||(T−F0)−(S−F0)||²=||T−S||²`，仅改称“创新量/反事实残差”没有改变损失。这里是简单代数，不是新理论。

出处及实际阅读范围见[note67](../../5.prior-art-survey/docs/67-distillation-targets-and-state-operator-support.md)。下一问题需区别于原GT重建和已知KD，同时尊重R74/R82等局部STOP，不通过给教师、损失或旧H改名复活旧试验。
