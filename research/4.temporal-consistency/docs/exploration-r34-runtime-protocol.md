# R34最终权重的CPU软件流水线成本对照

登记：2026-11-12T06:16:50+08:00；作者：Codex  (sqs_A100)。目的：用实际完整计算路径补充卷积MAC，检验低维状态是否换得总成本收益；不是新模型/质量参数网格，也不是手机测量。

仅在两个seed三个模型均正常完成4000更新后启动，严格加载各最终权重，不测中间权重或按DEV质量选择。固定R30 bear_photons20前16帧480×854作为性能工作负载，不读取GT/计算质量、不读独立test-dev。两条Torch线程/一条OpenCV线程、FP32、无CUDA/编译/额外teacher。每个映射每轮用前4帧warmup后reset，再计16帧完整过程。三轮执行顺序轮转并反向，报告所有轮次/seed、均值/中位数及范围，不以最快一次代表性能。共享服务器结果不可作手机延迟/功耗保证。

七映射：original raw/self-motion post .5；独立spatial raw/self-motion post .5；causal自身current raw/self-motion post .5、actual raw。self-motion使用自身前一raw增强、同half-FB/warp/blend，有限一阶、首帧原样、不额外跑teacher。所有模型逐帧同batch1，不以batch16加速某个控制。参数计预测器全部参数，过去状态统计真实storage/独占RGB数组；raw stateless路径在调用前reset，不能偷带上一轮状态。

计时从内存中的low帧tensor准备、CNN、clamp/布局转换、必要motion/warp/blend，到内存输出完成；分项与全循环同时记录。文件IO、模型加载、最终stack/指标/保存不计；循环会保留输出引用，明确该小量开销及离线输出内存不等于在线历史state/完整峰值。记录是否有限及输出shape，不做逐bit优化器/CPU回放扩展。不能用低卷积数或更小state直接推断更快。

该成本阶段与R34既定训练/120行质量评价独立，不改变其预算、门槛、最终权重选择或当前活进程；全部质量—时序结论仍须等双seed完整DEV评价。若训练失败不测不完整权重；若只有CPU数据，手机栏继续未测。协议、代码、原始轮次、汇总、过程、报告分别小步提交，阶段推送。
