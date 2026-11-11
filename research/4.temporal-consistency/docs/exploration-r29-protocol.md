# R29 静态重复观察的潜特征矩分解

创建：2026-11-12T01:35:21.895762+08:00；作者：Codex  (sqs_A100)；原因：跨帧乘积不自动消除head非线性/编码偏差，用既有重复观察分离条件均值、方差和协方差，而不是扩大模型网格。

零新训练、零新视频，CPU两线程/batch8。固定R19既有16条件（Kodak TRAIN01/06两图，photons20/80，rho0/.9，JPEG开/关，64对draws，各128×128），全部NPZ/数组SHA核验，不重采噪声。五套权重与R27完全相同：官方、R26 plain Charb/MSE双seed4000。模型冻结，仅提取head.block1和body.block1(head(x))两个FST之前的12通道特征；共160潜特征统计行，非160视频/质量配置。

参考u0为同一无噪声低照度信号s=(target^2.2)*.35按原encode(clip/uint8/JPEG)编码后的特征，不是亮GT特征，也不是没有输入偏差的理想潜变量。保留编码和非线性导致的均值偏移。空间margin8沿用R19，所有矩先对64draws逐像素取经验总体均值/方差（ddof0）/配对协方差，再汇总空间/通道。模型FP32提取，矩和理想q=a*u²+b用FP64；明确这不是FP32整网重放。

对当前u、过去v，计算mu_u/mu_v、var_u/var_v、cov_uv；自平方偏差分解为a*(mu_u²−u0²)+a*var_u；cross偏差为a*(mu_u*mu_v−u0²)+a*cov_uv。还记录postmean与pre_square的经验偏差。每个控制直接64draws均值与公式逐像素闭合，尺度残差<=1e-10；postmean−cross=a/2*E[(u−v)²]及pre_square−cross=a/4*E[(u−v)²]同门槛。拒绝用分量绝对值比例作为可加因果归因。

所有current编码/特征在rho间应逐像素SHA一致，共80特征重放；u0只随图/JPEG/权重/节点变化（40个缓存），不随photons/rho变化。保存带符号均值及RMS/绝对值/latent bias均方，原始潜特征不进Git。完整报告全部权重、两个节点和16条件；无flow/插值，因而隔离R22空间重采样影响。零CUDA初始化、无数据/权重安装，最终保留素材未读。

如果cross降低某一潜层偏差，也不能推断最终RGB增强优于强post；head/body均值偏移可能存在，attention/tail仍非线性。ρ/JPEG条件描述只针对固定合成缓存，不直接代表自然视频噪声。此为基础矩恒等式诊断，非新理论/新训练方法。R28算法预算不变，继续等待/执行其严格CUDA前置门槛。
