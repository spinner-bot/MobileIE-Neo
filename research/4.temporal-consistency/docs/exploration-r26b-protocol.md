# R26b 训练后骨干的单帧加后处理强控制

创建：2026-11-12T00:13:31.786532+08:00；作者：Codex  (sqs_A100)；原因：R26全表中MSE前聚合的current替代提升超过actual history，必须排除训练分布改变所带来的单帧映射收益。

零新训练/权重/数据，pre_fixedhalf/post_fixedhalf的Charb/MSE双seed1000/4000全部16检查点，仅取同一已训练backbone运行单帧F(current)。不调用过去输入或聚合分支；当前极限下pre current-current减为0、post同值平均即原值，函数应严格回放原current替代结果。所有4075参数仍对应原训练网络，不把这种部署映射当新结构。

原46探索条件、全部reset，四输出raw/recursive half01/finite postwarp.1/.5。两架构每个2944、总5888逻辑行：2944 raw/recursive原current参考完整逐帧/逐片重放，2944新增finite post控制；预期736真实神经序列推理、4096指标计算/1792精确alias。所有原source相同GPU后端（CUDA FP32/batch1/deterministic/TF32沿用R26），不能混CPU数字。每次启动需新两次空闲预检，只用同组空闲卡。

原current的raw/half01逐帧及逐片指标最大差预设<=1e-10，firstframe<=1e-4灰度级；失败保留、不放宽门槛。完整报告双loss/seed/固定checkpoint/四输出及reset，不按最优结果删减；旧权重SHA/当前形状及全部alias审计。若浮点路径不同，先查相同输入、算子顺序和后端，不直接假定数学恒等即数值相同。

这组是必要更强的单帧/训练增强控制：单帧backbone每输出一次forward，可缓存过去输出再warp；实际pre融合使用teacher输入历史，post需额外aligned-input映射。共享实验耗时不当独立pipeline/手机延迟。性能即使提高，也不能归为推理时真实历史利用或新颖性。最终保留素材未读，主研究继续active。
