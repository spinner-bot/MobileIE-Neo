# R19 已知时间信息（进行中）

创建：2026-11-11T21:27:10.998954+08:00；作者：Codex  (sqs_A100)；原因：接续R16/R18反证，在已知完美对应下分离可用信息与模型利用。

固定2训练Kodak patch、两photon/两JPEG/两rho共16条件64对；Poisson thinning/条件Gaussian耦合使current跨rho精确不变，后处理实际相关另测。四CPU fixture通过，rho1编码后精确相同、固定大样本边际矩/相关、方差恒等式与同current均核验。

九个固定SHA权重，current/preaverage与两时域类learned-history共22路径，352行；只诊断不新训练，不声称真实运动性能或新方法。配置/源码/检查已分别提交；计划独立空闲GPU2两次预检，不设墙钟截止。R16/R18阶段SSH推送已成功fbb0f4e，继续逐文件提交并在本阶段完成后推送。
