# 监督目标、条件无偏与有限预算

创建：2026-11-11T23:56:37.412147+08:00；作者：Codex  (sqs_A100)；原因：R26配对Charb/MSE训练完成，厘清目标统计、空间相关性及优化耦合，避免简单宣称L2更正确。

[Noise2Noise ICML2018一手正文](https://proceedings.mlr.press/v80/lehtinen18a/lehtinen18a.pdf) §2将L2/L1与条件均值/中位数联系，并写出噪声目标的条件均值要求；§3.1考察空间相关噪声下的收敛，§3.2提醒饱和裁截可能破坏均值条件。本次只精读这些段落，[阅读范围](../process/noise2noise-objective-reading/source-reading.json)已保存，未复现其模型或视频性能。

## 本项目的分析与边界

Charbonnier eps=.001是平滑L1，不能精确称为中位数优化。其标量条件一阶式为E[(z-Y)/sqrt((z-Y)²+1e-6)|X]=0；MSE对应z=E[Y|X]。有限容量、4000步和Adam/clip不保证到达理论最优；R26使用配对GT监督，不是Noise2Noise训练，也未先证明本项目target噪声满足条件无偏。

应区分两种假设：blind-spot利用空间邻域预测某像素时，邻域噪声相关会影响屏蔽的意义；Noise2Noise的条件均值条件并不等于“每帧内部所有像素噪声独立”。空间相关不自动使所有Noise2Noise训练无效。

若固定线性W只依赖输入，且E[e_target|input]=0，则E[W e_target|input]=0；但还需正确的干净信号对应，不能由零噪声均值推出W y_previous=y_current。若flow同时由有噪输入/目标估计，W也依赖target噪声，便不能直接把W移到条件期望外。clip/JPEG/非线性处理又可能改变均值。以上是本项目的假设检查，不是新的自监督证明。

R26实际梯度/clip耦合很明显：各Charb运行约95%更新触发clip1，原尺度MSE约11%–14%。所有初始同映射输出完全相同，Charb模型及Adam的48k控制链已核验，差异仍是整个目标/优化设定而非纯梯度方向。不能仅凭MSE训练损失数值更小，或PSNR由MSE定义，就宣称它解决了闪烁/实现架构创新。

主评价固定两checkpoint/双seed/所有46条件和MSEplain+经典post强参考，采用共同后端；PSNR、SSIM、edge、GT估计flow时序误差需同时报告。R26仍是探索性强控制，最终测试未读，后续结构设计须有额外证据与重复风险核验。
