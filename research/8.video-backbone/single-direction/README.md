# S1：单向传输与贡献者离散度

2026-11-20；Codex  (sqs_A100)。P1关闭后的独立信息问题：单向field能否同时携带历史值和目标碰撞信息，保留原F0恢复能力并减少field需求？前向归一化和flow方差已有先例，此处没有确立创新或加速。

[直接先例](../../5.prior-art-survey/docs/74-one-direction-transport-and-ambiguity-priors.md)、[首次执行前固定协议](protocol.md)、[算子](transport.py)、[唯一筛查](probe.py)、[26具名源/输入冻结](process/source-freeze001.json)。仅原训练窗口/双冻结F0，64NN、48双向DIS对和48共同GT-derived FB评分对，零优化/GPU/DEV/最终。

保留original_FB、取消FB的单backward、归一化push和目标贡献者flow离散度push；新候选唯一为最后一条，不从控制里选冠军。所有路径共享已计算fields，不能将本次运行计时称单向或手机速度。需完整原门槛才能另登记完整DEV/成本。

原V1/T1/T2/P1与旧R编号STOP不变。指定作者/main/no-ff/密集提交、自有data、7号与其他worktree、原84值/PDF020及独立最终封存保持。
