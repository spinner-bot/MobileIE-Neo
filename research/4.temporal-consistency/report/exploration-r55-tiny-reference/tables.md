# 固定696参数公开空间参考：完整迁移结果

实际模型696参数，原LOL-v1单一公开checkpoint，无项目训练/虚构seed。
20配对质量行/896逐帧；另6手机描述行/1904输出帧。全13CNN序列/1400真实输入帧，人工门槛另6张。
MobileIE为R44 square_mse8000及R45 RGB4000固定参考。预训练/数据/预算不同；不是同任务同预算架构因果、独立最终或作者原表复现。
不GT-Mean/自集成；时间列为GT变化残差而非时延；所有胜负保留。

|域|权重/seed|方法|参数|PSNR|SSIM|时间误差↓|edge↓|
|---|---|---|---:|---:|---:|---:|---:|
|davis_train_pilot|published|Nano_raw|696|20.282069|0.481416|18.464234|5.801441|
|davis_train_pilot|published|Nano_recursive_self05|696|21.932863|0.618069|10.454472|4.273360|
|paired_real_ND_DEV|published|Nano_raw|696|21.393795|0.841423|4.547059|1.816781|
|paired_real_ND_DEV|published|Nano_recursive_self05|696|21.387387|0.877235|2.963915|1.533629|
|davis_train_pilot|seed0|MobileIE_raw|4075|26.475237|0.714517|10.101737|4.304666|
|davis_train_pilot|seed0|MobileIE_fixed_05|4075|26.800044|0.768807|6.545945|3.809294|
|davis_train_pilot|seed0|MobileIE_rgb|4116|27.172215|0.759827|7.305598|3.867806|
|paired_real_ND_DEV|seed0|MobileIE_raw|4075|27.252453|0.909500|3.728773|1.535234|
|paired_real_ND_DEV|seed0|MobileIE_fixed_05|4075|27.292047|0.920910|2.605355|1.395794|
|paired_real_ND_DEV|seed0|MobileIE_rgb|4116|27.387785|0.921679|2.790698|1.400289|
|davis_train_pilot|seed1|MobileIE_raw|4075|26.456334|0.717926|9.699145|4.249495|
|davis_train_pilot|seed1|MobileIE_fixed_05|4075|26.612977|0.766202|6.393445|3.832977|
|davis_train_pilot|seed1|MobileIE_rgb|4116|27.052034|0.760930|6.975614|3.856254|
|paired_real_ND_DEV|seed1|MobileIE_raw|4075|27.004095|0.913022|3.576601|1.499359|
|paired_real_ND_DEV|seed1|MobileIE_fixed_05|4075|26.923483|0.923179|2.534350|1.374671|
|paired_real_ND_DEV|seed1|MobileIE_rgb|4116|27.079505|0.924536|2.676844|1.372133|

|输入H×W|实际Conv MAC|
|---|---:|
|288×512|93339648|
|360×640|145843200|
|480×854|259479360|
|1080×1920|1312588800|

仅已执行Conv形状MAC，解析先验/归一/规约/激活/flow/warp/内存/IO未计；不是完整FLOPs、实测时延或手机性能。
手机仅同三已见DEV，无GT，不按亮度/平滑代理排名。原始FP32和固定全帧预览另存；独立新方法仍未建立。
