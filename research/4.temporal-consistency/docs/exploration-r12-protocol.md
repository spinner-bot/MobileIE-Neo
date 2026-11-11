# R12 已发表五帧视频去噪强对照

创建：2026-11-11T18:56:00.507628+08:00；作者：Codex  (sqs_A100)；原因：补充实际可运行的已发表视频方法，避免仅在自设轻量模块之间比较；R11完整评价仍在运行。

## 固定方法及解释边界

FastDVDnet官方commit c8fdf6182a0340e89dd18f5df25b47337cbede6f，MIT代码models.py/fastdvdnet.py原字节复制并保留LICENSE。固定官方model_clipped_noise.pth（Git blob413d19765b821355e31906ca16332006d8d3a4eb、9971551字节），另记SHA256。代码及README已经核查，官方test脚本会新增AWGN，不能直接拿它处理本项目退化；适配器调用原网络/推理函数，不加新噪声、不重训练。

仅对R9full1e-3两seed的clipped增强RGB作去噪；不因R11结果选择更有利底座。noise map固定sigma/255，sigma=5/15/30全部报告，不看dev/test GT选择sigma。高斯clipped预训练与泊松/JPEG/ND/增强噪声不匹配，结果是方法迁移对照，不是原论文标准DAVIS基准复现，也不是已经完成LLVE论文强基线。

五帧中心窗口需两帧未来，明确至少两帧输入等待；不同于本项目零未来候选。比较3种调用：window无事件分段、window按固定输入检测分段并在片段边界reflect、five_current五份当前帧输入。后者是同网络去除真实时间信息的空间对照，不称原论文单帧模型。分段可能使用未来窗口内检测事件，仍为两帧未来方法，不能称因果reset。

## 验证与预算

先运行固定可证伪smoke：8帧Kodak训练01衍生256×192缓慢平移，独立clipped AWGN sigma25、seed12。严格加载所有参数，预设输出有限、去噪PSNR比输入高至少1dB；适配器window与官方denoise_seq_fastdvdnet归一化像素最大差≤1e-6。同帧复制/reflect/分段边界另作独立路由fixture。若失败，保留并定位，不降低预设容差。

随后原38探索+4dev resize+同4dev native共46条件，底座2seed×(原始raw1+三sigma×三调用9)=920行（其中92raw重评、828去噪）。保留全部PSNR/SSIM/edge/GT残差时间误差、逐帧/逐片和事件前后，不增添本方法后处理。底座raw重放已有评价最大指标差≤1e-6。future window与five_current必须配对比较，不将AWGN空间去噪当时间贡献。

固定无训练、无墙钟截断。运行前每GPU任务重新两次空闲预检，仅同组最多4空闲卡，计划仅2/3；源码、源清单/权重SHA、测试、smoke、评价、过程与报告分别小步提交。权重/cache像素不入Git。完整串行端到端成本与手机测量另补，GPU forward不能充当手机实时性。
