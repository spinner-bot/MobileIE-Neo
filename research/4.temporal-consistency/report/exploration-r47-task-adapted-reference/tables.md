# R47 同资料任务适配视频参考：完整结果

两个seed各8000更新，总16000。40新CPU质量/CNN行，1792逐帧行；不按胜负筛选参考。
FastDVDnet原五帧网络直接低光到GT，2479096参数；MobileIE为最终8000强空间底座，4075参数。
前者window未来等待2帧、five_current为同权重五份当前；不同预训练/容量/BN/训练FLOPs，且公开预训练与三DEV视频有来源重叠。

|域|seed|方法|PSNR|SSIM|时间↓|edge↓|
|---|---:|---|---:|---:|---:|---:|
|davis_train_pilot|0|FastDVD_task_window|25.430848|0.755591|6.648250|3.724116|
|davis_train_pilot|0|FastDVD_task_five_current|25.921951|0.755817|7.962015|3.817026|
|paired_real_ND_DEV|0|FastDVD_task_window|21.953304|0.912240|3.603707|1.439149|
|paired_real_ND_DEV|0|FastDVD_task_five_current|24.539476|0.917848|3.455530|1.430844|
|davis_train_pilot|0|MobileIE_raw|26.475237|0.714517|10.101737|4.304666|
|davis_train_pilot|0|MobileIE_fixed_025|26.994866|0.750926|7.895790|3.944723|
|davis_train_pilot|0|MobileIE_fixed_05|26.800044|0.768807|6.545945|3.809294|
|davis_train_pilot|0|MobileIE_rgb|27.172215|0.759827|7.305598|3.867806|
|paired_real_ND_DEV|0|MobileIE_raw|27.252453|0.909500|3.728773|1.535234|
|paired_real_ND_DEV|0|MobileIE_fixed_025|27.367955|0.919433|2.993846|1.426200|
|paired_real_ND_DEV|0|MobileIE_fixed_05|27.292047|0.920910|2.605355|1.395794|
|paired_real_ND_DEV|0|MobileIE_rgb|27.387785|0.921679|2.790698|1.400289|
|davis_train_pilot|1|FastDVD_task_window|22.829883|0.724007|7.378442|4.034454|
|davis_train_pilot|1|FastDVD_task_five_current|23.011624|0.730543|9.229979|4.180392|
|paired_real_ND_DEV|1|FastDVD_task_window|23.356367|0.907679|3.498059|1.450089|
|paired_real_ND_DEV|1|FastDVD_task_five_current|25.122806|0.914461|3.269119|1.433638|
|davis_train_pilot|1|MobileIE_raw|26.456334|0.717926|9.699145|4.249495|
|davis_train_pilot|1|MobileIE_fixed_025|26.870053|0.751296|7.640599|3.936582|
|davis_train_pilot|1|MobileIE_fixed_05|26.612977|0.766202|6.393445|3.832977|
|davis_train_pilot|1|MobileIE_rgb|27.052034|0.760930|6.975614|3.856254|
|paired_real_ND_DEV|1|MobileIE_raw|27.004095|0.913022|3.576601|1.499359|
|paired_real_ND_DEV|1|MobileIE_fixed_025|27.066325|0.922301|2.886506|1.398058|
|paired_real_ND_DEV|1|MobileIE_fixed_05|26.923483|0.923179|2.534350|1.374671|
|paired_real_ND_DEV|1|MobileIE_rgb|27.079505|0.924536|2.676844|1.372133|

固定有限预算适配不等于原论文最优或充分收敛；同更新数不是等计算量；CPU任务耗时不是手机实测。无独立最终或自然夜间结论，本表补足公开视频参考，不建立本项目创新。
