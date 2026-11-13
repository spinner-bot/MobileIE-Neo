# R39：固定空间anchor在既有配对实拍DEV上的共同CPU转移

登记：2026-11-12T07:51:31+08:00；作者：Codex  (sqs_A100)。R38三手机无GT固定预览显示截断/色偏，不能由合成DEV高分推断适用于真实输入。原始上游LLEData已核查为PIL RGB、float32/255；本项目手机缓存也是RGB，无BGR互换依据。R37仍按原协议完成评价，不因这些观察改变训练或门槛。

只读已反复用作DEV的R10 index中全部四个native_crop（video_15, video_47, video_91, video_121），各16帧288×512，不触及未读TEST3/Kodak保留/test-dev。保留原配对输入/GT/评价对应、不加退化或事件reset。统一CPU2/OpenCV1、逐帧B1 FP32，比较五个固定anchor：官方LOL、R26 plain MSE/final4000 seed0/1、R34 original/final4000 seed0/1。每个raw、finite self-post .5、recursive self-post .5，共60质量行/20次CNN序列。所有权重与四片均在看本轮结果前固定，不按手机观感选择seed。

post运动均由各anchor自身相邻raw估计half-FB，finite用上一raw，recursive用上一融合，无效回退当前。统一compute PSNR/SSIM/固定参考对应残差temporal/edge，不拿旧CUDA指标直接与新CPU微小差异排名。此处R26和R34的数据/优化/batch都不同，整体训练方案转移可比较，不能将差异孤立归于一个因素；官方权重只一份，不伪造双seed。

完整60行后分别列每anchor和两训练方案的双seed均值、逐片配对。正式问题是R34在合成DEV的约2.57dB raw收益是否在现有配对实拍上保持方向，而非挑个别场景宣布泛化。没有新模型/参数网格、训练或CPU位级复现。若明显域代价，下一研究先修正/检验数据与光度表征假设，暂不在失败底座上扩大时间架构；若转移良好，仍保留手机非配对饱和问题作为独立限制。四片白天ND实拍不等同自然夜景，不是最终独立确认或手机速度。
