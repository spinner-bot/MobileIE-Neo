# R26监督目标控制过程记录

创建：2026-11-12T00:03:31.017038+08:00；作者：Codex  (sqs_A100)；原因：48000更新完整结束，评价仍进行，分阶段记录事实与未定结论。

先固定plain/pre.5/post.5×Charb/MSE双seed4000及完整8832行评价，六模型/Adam每seed独立，仅共享一次固定teacher对应。日志目录缺失导致首次shell启动在guard/Python前退出，0更新留档；补目录后GPU1占用被guard拒绝。seed1 GPU3、seed0 GPU0都经新两次空闲检查，完整完成4000，不设墙钟截断。

初始current函数差0、Charb/MSE同映射输出完全一致；未按初始norm/cosine或dev缩放loss。所有4000当前任务上下文与R14一致，两seed各16000历史SHA完全匹配。12组48000更新、48检查点/Adam步数/有限性/前缀/386训练源SHA通过；Charb24检查点全部model/Adam与旧R14/R21逐tensor完全一致。每1000 marker/prefix及完成trace/日志/源完整性分别提交，训练资源专段2成功1拒绝、峰值2同组GPU。

训练审计显示Charb约95%clip、MSE约11%–14%，因此是整个目标/优化配置而非纯方向实验。报告0037完整列训练事实；Noise2Noise §2/相关噪声及裁截段落已核验，未把理论均值最优等同短预算视频更好。

训练退出后新预检GPU0/2/3为空闲，启动三组共同CUDA评价plain/pre/post。全部目标/1000/4000/seed/reset/历史与plain新MSE+finite经典后处理不删减。评价未结束，只提交immutable启动配置/检查点清单和guard快照；完整结果完成后另报，最终保留素材未读，论文目标active。
