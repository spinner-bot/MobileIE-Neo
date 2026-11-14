# R56 固定8000更新的微型公开参考：全部结果

原Nano696参数/BN保持，两个seed各8000新更新，原MixedPool/四目标/raw MSE/Adam1e-4/4000重置。
同目标/更新数不等于同预训练、MBR参数化、训练计算或充分收敛；这是既有方法的必要适配对照，不是新方法。
新CPU40配对质量/1792逐帧、12手机描述/3808输出帧，26CNN序列/2800真实输入帧。全部结果保留，无最优seed/中间点。
手机三已见DEV无GT，不能作质量/真闪烁排名。R53视频参考另按其完整固定链收齐，不以部分值参入。

|域|权重/seed|方法|参数|PSNR|SSIM|GT变化时间误差↓|edge↓|
|---|---|---|---:|---:|---:|---:|---:|
|davis_train_pilot|seed0|Nano_adapted_raw|696|22.474246|0.587120|13.172008|4.884313|
|davis_train_pilot|seed0|Nano_adapted_recursive_self05|696|23.386295|0.684024|8.057163|4.096925|
|paired_real_ND_DEV|seed0|Nano_adapted_raw|696|24.342454|0.888832|3.708536|1.606046|
|paired_real_ND_DEV|seed0|Nano_adapted_recursive_self05|696|24.164243|0.904520|2.576645|1.486460|
|davis_train_pilot|seed0|MobileIE_raw|4075|26.475237|0.714517|10.101737|4.304666|
|davis_train_pilot|seed0|MobileIE_fixed_05|4075|26.800044|0.768807|6.545945|3.809294|
|davis_train_pilot|seed0|MobileIE_rgb|4116|27.172215|0.759827|7.305598|3.867806|
|paired_real_ND_DEV|seed0|MobileIE_raw|4075|27.252453|0.909500|3.728773|1.535234|
|paired_real_ND_DEV|seed0|MobileIE_fixed_05|4075|27.292047|0.920910|2.605355|1.395794|
|paired_real_ND_DEV|seed0|MobileIE_rgb|4116|27.387785|0.921679|2.790698|1.400289|
|davis_train_pilot|seed1|Nano_adapted_raw|696|22.204272|0.580965|13.130837|4.831502|
|davis_train_pilot|seed1|Nano_adapted_recursive_self05|696|23.032550|0.677652|8.039061|4.084921|
|paired_real_ND_DEV|seed1|Nano_adapted_raw|696|23.901657|0.886515|3.708394|1.609773|
|paired_real_ND_DEV|seed1|Nano_adapted_recursive_self05|696|23.742888|0.901994|2.581099|1.494963|
|davis_train_pilot|seed1|MobileIE_raw|4075|26.456334|0.717926|9.699145|4.249495|
|davis_train_pilot|seed1|MobileIE_fixed_05|4075|26.612977|0.766202|6.393445|3.832977|
|davis_train_pilot|seed1|MobileIE_rgb|4116|27.052034|0.760930|6.975614|3.856254|
|paired_real_ND_DEV|seed1|MobileIE_raw|4075|27.004095|0.913022|3.576601|1.499359|
|paired_real_ND_DEV|seed1|MobileIE_fixed_05|4075|26.923483|0.923179|2.534350|1.374671|
|paired_real_ND_DEV|seed1|MobileIE_rgb|4116|27.079505|0.924536|2.676844|1.372133|
|davis_train_pilot|published|Nano_published_raw|696|20.282069|0.481416|18.464234|5.801441|
|davis_train_pilot|published|Nano_published_recursive_self05|696|21.932863|0.618069|10.454472|4.273360|
|paired_real_ND_DEV|published|Nano_published_raw|696|21.393795|0.841423|4.547059|1.816781|
|paired_real_ND_DEV|published|Nano_published_recursive_self05|696|21.387387|0.877235|2.963915|1.533629|

网络图/参数形状未变，Conv MAC可沿R55同图形状计数；解析先验/归一/pointwise/flow/warp/内存/IO未计，非手机速度。
