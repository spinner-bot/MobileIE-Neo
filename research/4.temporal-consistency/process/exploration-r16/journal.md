# R16 多尺度时域机制（进行中）

创建：2026-11-11T20:28:29.512739+08:00；作者：Codex  (sqs_A100)；原因：接续已完成R14/R15，按固定协议检验空间层级与历史独立作用。

标准encoder-decoder参考，参数24955（以实际config为准），三路同容量：spatial/raw_history/aligned_history。CPU四检查通过，初版scalar警告保留，detach修订后复核；不放宽初始官方输出2e-5门槛。训练原R14全部4000步监督/ROI/合成上下文逐项相等，双seed各4000，共24000新更新；新构建骨干初始化，不续选最佳权重。

/data挂载复查，0–3为空闲，4–7其他用户使用中。计划GPU2跑spatial/raw共16000、GPU3跑aligned共8000；每项仍由guard独立两次空闲采样决定是否执行。无墙钟截断，1000步checkpoint/trace前缀审计后小步提交。
