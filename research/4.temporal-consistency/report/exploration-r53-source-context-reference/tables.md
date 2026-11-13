# R53 源上下文适配：完整有限预算参考

新训练双seed各8000更新；新CPU40行/40CNN/1792逐帧。全部结果保留。
R53保留R47四个监督目标，但额外看到两侧源输入；不能将差异仅归因为消除重复，也不是等信息控制。
FastDVDnet 2479096参数、公开AWGN预训练与三DEV源重叠、window未来2帧；MobileIE4075参数。
同更新数不等于同计算量；有限适配不证明充分收敛。时间列是GT变化残差指标，不是延迟。

|域|seed|方法|PSNR|SSIM|时间误差↓|edge↓|
|---|---:|---|---:|---:|---:|---:|
|davis_train_pilot|0|source_context_window|25.332325|0.759756|6.646311|3.699978|
|davis_train_pilot|0|source_context_five_current|25.435312|0.752920|8.557148|3.912939|
|paired_real_ND_DEV|0|source_context_window|22.349896|0.910532|3.496429|1.449335|
|paired_real_ND_DEV|0|source_context_five_current|24.552092|0.917105|3.276754|1.425497|
|davis_train_pilot|0|short_context_window|25.430848|0.755591|6.648250|3.724116|
|davis_train_pilot|0|short_context_five_current|25.921951|0.755817|7.962015|3.817026|
|paired_real_ND_DEV|0|short_context_window|21.953304|0.912240|3.603707|1.439149|
|paired_real_ND_DEV|0|short_context_five_current|24.539476|0.917848|3.455530|1.430844|
|davis_train_pilot|0|MobileIE_raw|26.475237|0.714517|10.101737|4.304666|
|davis_train_pilot|0|MobileIE_fixed_025|26.994866|0.750926|7.895790|3.944723|
|davis_train_pilot|0|MobileIE_fixed_05|26.800044|0.768807|6.545945|3.809294|
|davis_train_pilot|0|MobileIE_rgb|27.172215|0.759827|7.305598|3.867806|
|paired_real_ND_DEV|0|MobileIE_raw|27.252453|0.909500|3.728773|1.535234|
|paired_real_ND_DEV|0|MobileIE_fixed_025|27.367955|0.919433|2.993846|1.426200|
|paired_real_ND_DEV|0|MobileIE_fixed_05|27.292047|0.920910|2.605355|1.395794|
|paired_real_ND_DEV|0|MobileIE_rgb|27.387785|0.921679|2.790698|1.400289|
|davis_train_pilot|1|source_context_window|22.582077|0.727464|7.552979|3.940522|
|davis_train_pilot|1|source_context_five_current|23.996981|0.746975|8.526232|3.904786|
|paired_real_ND_DEV|1|source_context_window|22.963812|0.908426|3.529856|1.429326|
|paired_real_ND_DEV|1|source_context_five_current|24.871705|0.912118|3.315648|1.427900|
|davis_train_pilot|1|short_context_window|22.829883|0.724007|7.378442|4.034454|
|davis_train_pilot|1|short_context_five_current|23.011624|0.730543|9.229979|4.180392|
|paired_real_ND_DEV|1|short_context_window|23.356367|0.907679|3.498059|1.450089|
|paired_real_ND_DEV|1|short_context_five_current|25.122806|0.914461|3.269119|1.433638|
|davis_train_pilot|1|MobileIE_raw|26.456334|0.717926|9.699145|4.249495|
|davis_train_pilot|1|MobileIE_fixed_025|26.870053|0.751296|7.640599|3.936582|
|davis_train_pilot|1|MobileIE_fixed_05|26.612977|0.766202|6.393445|3.832977|
|davis_train_pilot|1|MobileIE_rgb|27.052034|0.760930|6.975614|3.856254|
|paired_real_ND_DEV|1|MobileIE_raw|27.004095|0.913022|3.576601|1.499359|
|paired_real_ND_DEV|1|MobileIE_fixed_025|27.066325|0.922301|2.886506|1.398058|
|paired_real_ND_DEV|1|MobileIE_fixed_05|26.923483|0.923179|2.534350|1.374671|
|paired_real_ND_DEV|1|MobileIE_rgb|27.079505|0.924536|2.676844|1.372133|

这是既有视频模型的训练设计控制，不建立本项目新颖性、独立最终/自然夜间泛化或手机实测结论。
