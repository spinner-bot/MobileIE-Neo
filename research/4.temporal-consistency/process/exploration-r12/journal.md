# R12 已发表五帧去噪对照

创建：2026-11-11T19:14:18.850601+08:00；作者：Codex  (sqs_A100)；原因：920评价及官方等价smoke完成。

GitHub连接器读取官方固定commit c8fdf6182a0340e89dd18f5df25b47337cbede6f模型/推理/测试脚本及MIT许可。原字节vendor LICENSE/models.py/fastdvdnet.py；Git blob SHA及bytes一致。连接器拒绝二进制，按官方固定raw URL下载9971551字节clipped checkpoint，SHA256 8118974ac7defaa5037f73caf87e0cb53efcfa49ae77d55c05ab187f59e55949，Git blob413d19765b821355e31906ca16332006d8d3a4eb通过。官方测试脚本会新增AWGN，适配器直接调用推理而不新增噪声。

先登记sigma5/15/30全部报告，固定R9full1e-3两seed底座，不依R11选底座。新增3项路由检查通过，覆盖reflect、未来数、分段边界与singleton。固定训练Kodak01衍生8帧clipped AWGN25 smoke：PSNR20.30510→29.77754（+9.47244），与官方归一化像素最大差0，预设1e-6和最少1dB门槛通过。不是低光迁移证据。

46条件×2seed×[raw+3sigma×3调用]=920行，base重放92行最大指标差0。window两帧未来，window_segmented仅片段内reflect，five_current五份自身；两个seed的全部window索引一致，最大未来2，five_current为0，事件分段没有跨界。原生/resize仍共享4dev，不是新物理源。

native sigma5 window相对five_current：PSNR+.04473（3/4正）、SSIM+.00748（4/4正）、时间误差-.28142及edge-.11510（均4/4改善），显示真实五帧信息窗口。sigma15相对raw：PSNR+.10303（3/4）、SSIM+.02798（4/4）、时间误差-1.46890（4/4）；但相对five_current PSNR-.01751，不能将总收益全归时间信息。resize dev同sigma15明显损失SSIM（.89835→.83618），time2.46665→2.55125也变差。不是跨域统一方法。

模型2479096参数，窗口需未来2帧；当前tiny零未来不能直接混同。480270串行GPU阶段中位base.6147ms、denoiser9.7322ms，512288 base.6470/denoiser10.6901；包括Python/分配，但排除数据/传输/指标/CPU完整流程，且共享A100，不宣称手机速度或与R4完整流水线直接比值。

3GPU任务全部成功，仅2/3卡并发2，225启动源码哈希匹配，共468.153进程GPU秒含CPU。权重载入前核验vendor/weight SHA，文件、原始评价、统计、图、过程与报告分别提交。R13转向训练first-self占比的受控诊断，新增前置历史而不改当前监督，空间1000步重放已为0差。
