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
|davis_train_pilot|StableLLVE_spatial_l1|0|raw|4316259|8000|7.656902|28.087254|0.775302|7.618724|3.646484|
|davis_train_pilot|StableLLVE_spatial_l1|0|recursive_self05|4316259|8000|7.631575|27.589501|0.786265|5.492944|3.562918|
|davis_train_pilot|StableLLVE_commute20|0|raw|4316259|8000|6.992263|25.368162|0.687459|7.743103|4.353732|
|davis_train_pilot|StableLLVE_commute20|0|recursive_self05|4316259|8000|6.911584|25.203931|0.717416|5.788697|4.160048|
|davis_train_pilot|StableLLVE_spatial_l1|1|raw|4316259|8000|7.639385|28.032949|0.773150|7.621716|3.656134|
|davis_train_pilot|StableLLVE_spatial_l1|1|recursive_self05|4316259|8000|7.613474|27.547429|0.784562|5.497065|3.569448|
|davis_train_pilot|StableLLVE_commute20|1|raw|4316259|8000|6.971648|25.245739|0.683761|7.874902|4.379476|
|davis_train_pilot|StableLLVE_commute20|1|recursive_self05|4316259|8000|6.894715|25.146531|0.715672|5.858975|4.169257|
|paired_real_ND_DEV|StableLLVE_spatial_l1|0|raw|4316259|8000|8.431688|28.732435|0.926278|3.235137|1.377298|
|paired_real_ND_DEV|StableLLVE_spatial_l1|0|recursive_self05|4316259|8000|8.429979|28.581811|0.927322|2.437496|1.322791|
|paired_real_ND_DEV|StableLLVE_commute20|0|raw|4316259|8000|7.765415|25.853189|0.894055|2.905363|1.681091|
|paired_real_ND_DEV|StableLLVE_commute20|0|recursive_self05|4316259|8000|7.651013|25.631601|0.894736|2.352879|1.649262|
|paired_real_ND_DEV|StableLLVE_spatial_l1|1|raw|4316259|8000|8.207790|28.419917|0.924460|3.697284|1.400215|
|paired_real_ND_DEV|StableLLVE_spatial_l1|1|recursive_self05|4316259|8000|8.247244|28.578084|0.928346|2.740803|1.304891|
|paired_real_ND_DEV|StableLLVE_commute20|1|raw|4316259|8000|7.675835|25.828274|0.893178|2.904683|1.697414|
|paired_real_ND_DEV|StableLLVE_commute20|1|recursive_self05|4316259|8000|7.561653|25.579416|0.894353|2.344268|1.662429|
