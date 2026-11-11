# R23 全局RGB偏差与空间误差分解

创建：2026-11-11T23:00:00+08:00；作者：Codex  (sqs_A100)；原因：R21动态与静态结果不一致，固定帧见整体颜色变化，先分离误差来源，不直接归因运动或增加模块。

零新训练/推理/像素，只复用R21定性清单中四native开发片段全部16帧、seed0/4000、noreset/raw的原始数组。Input/GT/预测均核验52个文件SHA；预测是在显示缩图/编码前保存的原值，不读取MP4计算指标。已有44序列及逐帧正式指标重放差0。本轮3强参考（R14、R20 postwarp.1/.5）及四R21模式current/causal，共11输出×4片=44序列/704帧。未选择最好seed，seed0是既有定性协议固定；不据此宣称双seed普遍性。

对每帧margin8的归一化RGB误差e=prediction-GT，逐通道空间均值μ，有MSE=mean_c(μ²)+mean((e-μ)²)。再把全局RGB项分为common-channel mean(μ)²及mean((μ-mean(μ))²)，这是Euclidean RGB正交分解，不是亮度/色度感知空间；前者不能直接称真实曝光误差。每序列mean_t mean_c μ_t²又等于mean_c(mean_t μ_t)²+mean_t,c(μ_t-mean_t μ_t)²，分别记录持续全局偏差和其时间波动，不把后者称主观闪烁或替代原运动补偿指标。

全部FP64，所有恒等式绝对残差<=1e-12。原MSE→每帧PSNR需与正式同CPUper_frame PSNR最大差<=1e-8dB，否则停止查错，不放宽门槛。保留所有帧/序列/通道均值。

固定比较每R21 current/causal对R14单帧、每causal对同模型current、每causal对R20 postwarp.5。保存signed ΔMSE=Δcommon-channel bias²+Δopponent-channel bias²+Δcentered error；比值允许负数或>1，不截为“百分比贡献”。GT均值移除后的中心化误差只作分析参考，不生成增强算法/新性能或因果归因；不改GT/原指标。

CPU/std数学分析，不占GPU。先确认质量差是全局映射还是空间项，再决定是否需要同接受域的GT估计motion分析；GT估计flow仍非真值、不作可部署结果。R22局部交换误差并不由本轮建立因果关系。最终保留素材未读，普通偏差分解不当论文创新。

修改：2026-11-11T23:02:50.051748+08:00；作者：Codex  (sqs_A100)；原因：首轮正式PSNR重放失败暴露口径误判，0统计行完成后先修订方案再重跑。

上述margin8分解与正式PSNR口径不匹配：实际exploration_r2_data.evaluate的PSNR在全帧计算，只有时序有效区裁边8。主分解改为全帧，以精确对应正式每帧MSE/PSNR；仍保持1e-12恒等式及1e-8dB重放门槛。失败资料完整保留，不改任何既有指标/GT/主评价、不放宽数值容差。
