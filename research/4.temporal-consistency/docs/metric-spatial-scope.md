# 当前指标的空间范围与比较边界

创建：2026-11-11T23:03:00+08:00；作者：Codex  (sqs_A100)；原因：R23首次PSNR重放失败暴露“所有指标都裁边8”的错误概括，以实际共享代码为准澄清，既有结果保持不变。

| 场景/指标 | 实际计算范围 | 代码入口 |
|---|---|---|
| 动态PSNR RGB | 全帧、全RGB，FP64 MSE，float[0,255]，不取整 | video_metrics.psnr；exploration_r2_data.evaluate |
| 动态SSIM RGB | 11×11 Gaussian局部统计，最终valid区裁边5；不是裁边8 | video_metrics.ssim |
| 动态edge MAE | 全帧灰度Sobel x/y，除8后取绝对差均值；保留OpenCV默认边界处理 | video_metrics.edge_error |
| 动态flicker/temporal_error | GT估计flow gather及reference mask，再将有效区四边各8置false；后者扣除同warp下GT帧间差 | exploration_r2_data.evaluate |
| R19/R19b/R21b静态条件矩 | margin8 RGB MSE/均值偏差/噪声draw总体方差；每draw PSNR再平均 | noise_moments_r6.rgb_statistics |
| R23误差分解 | 全帧RGB，与动态正式PSNR对齐；不是空间裁边8统计 | probe_rgb_bias_r23 |
| PNG/MP4展示 | 取整/缩图/编码仅显示，不参与数值评价 | preview_aggregation_r21 |

动态PSNR参考与静态蒙特卡罗PSNR的空间范围不同，且输入/退化/数据定义不同，不能直接拼成统一基准排名。静态诊断应比较相同条件、相同ROI、同batch及后端；动态主比较亦使用共同后端。R21已经测得CPU/GPU非零数值差，因此完整重算了同CPU强参考。

GT估计flow不是运动真值，时序误差不是感知闪烁的充分证明。R23的Euclidean RGB共同通道/对立通道分解也不是亮度/色度感知空间，GT均值移除仅用于误差分析。

本说明不修改旧GT、缓存、指标实现或已完成结果。R23误用margin8的首轮已在严格PSNR重放中失败，保留现场后改为正式全帧口径，容差未放宽。
