# 固定显示条件的感知补评

"ColorVideoVDP v0.5.7, 37.84 [pix/deg], Lpeak=200, Lblack=0.2, Lrefl=0.3979 [cd/m^2], (standard_fhd)"

Nominal30fps、原尺寸SDR、replicate padding；是显示假设下的预测，非实测手机/采集帧率、人类MOS或专门闪烁分数。
跨架构预训练/容量/目标/计算不等，FastDVDnet window等待2帧。Input/GT仅指标控制，无新增增强推理。

|域|模型|seed|输出|参数|项目更新|JOD↑|PSNR|SSIM|Etemp↓|edge↓|
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
|davis_train_pilot|Input|None|input|0|0|4.942827|—|—|—|—|
|davis_train_pilot|GT|None|identity|0|0|10.000000|—|—|—|—|
|davis_train_pilot|MobileIE|0|raw|4075|8000|7.040340|26.475237|0.714517|10.101737|4.304666|
|davis_train_pilot|MobileIE|0|recursive_self05|4075|8000|7.423528|26.800044|0.768807|6.545945|3.809294|
|davis_train_pilot|MobileIE|1|raw|4075|8000|7.110548|26.456334|0.717926|9.699145|4.249495|
|davis_train_pilot|MobileIE|1|recursive_self05|4075|8000|7.408986|26.612977|0.766202|6.393445|3.832977|
|davis_train_pilot|Nano|0|raw|696|8000|6.741845|22.474246|0.587120|13.172008|4.884313|
|davis_train_pilot|Nano|0|recursive_self05|696|8000|7.118368|23.386295|0.684024|8.057163|4.096925|
|davis_train_pilot|Nano|1|raw|696|8000|6.871522|22.204272|0.580965|13.130837|4.831502|
|davis_train_pilot|Nano|1|recursive_self05|696|8000|7.178260|23.032550|0.677652|8.039061|4.084921|
|davis_train_pilot|FastDVDnet|0|window|2479096|8000|7.431695|25.332325|0.759756|6.646311|3.699978|
|davis_train_pilot|FastDVDnet|1|window|2479096|8000|6.303795|22.582077|0.727464|7.552979|3.940522|
|davis_train_pilot|StableLLVE_clean|None|raw|4316259|0|6.261886|21.384658|0.622391|9.802266|4.667800|
|davis_train_pilot|StableLLVE_clean|None|recursive_self05|4316259|0|6.265936|21.540874|0.678547|6.667801|4.280807|
|paired_real_ND_DEV|Input|None|input|0|0|7.061170|—|—|—|—|
|paired_real_ND_DEV|GT|None|identity|0|0|10.000000|—|—|—|—|
|paired_real_ND_DEV|MobileIE|0|raw|4075|8000|8.414068|27.252453|0.909500|3.728773|1.535234|
|paired_real_ND_DEV|MobileIE|0|recursive_self05|4075|8000|8.373159|27.292047|0.920910|2.605355|1.395794|
|paired_real_ND_DEV|MobileIE|1|raw|4075|8000|8.552201|27.004095|0.913022|3.576601|1.499359|
|paired_real_ND_DEV|MobileIE|1|recursive_self05|4075|8000|8.512007|26.923483|0.923179|2.534350|1.374671|
|paired_real_ND_DEV|Nano|0|raw|696|8000|8.168777|24.342454|0.888832|3.708536|1.606046|
|paired_real_ND_DEV|Nano|0|recursive_self05|696|8000|8.114096|24.164243|0.904520|2.576645|1.486460|
|paired_real_ND_DEV|Nano|1|raw|696|8000|8.141369|23.901657|0.886515|3.708394|1.609773|
|paired_real_ND_DEV|Nano|1|recursive_self05|696|8000|8.079625|23.742888|0.901994|2.581099|1.494963|
|paired_real_ND_DEV|FastDVDnet|0|window|2479096|8000|7.680177|22.349896|0.910532|3.496429|1.449335|
|paired_real_ND_DEV|FastDVDnet|1|window|2479096|8000|7.257248|22.963812|0.908426|3.529856|1.429326|
|paired_real_ND_DEV|StableLLVE_clean|None|raw|4316259|0|6.802328|20.586070|0.852835|3.126002|1.863714|
|paired_real_ND_DEV|StableLLVE_clean|None|recursive_self05|4316259|0|6.631801|20.539086|0.854198|2.475855|1.814323|
