# R50 完整固定权重BN校准控制

全部六模型/120行/60CNN，0SGD；原R48门槛与止损不变。两seed差不是方差或显著性估计。

|模型|读出|域|seed|原PSNR|校准PSNR|ΔdB|原时间误差|校准时间误差|比|最差条件ΔdB|
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|square_mse|raw|davis_train_pilot|0|26.475237|26.481560|0.006323|10.101737|9.990305|0.988969|-0.038422|
|square_mse|raw|davis_train_pilot|1|26.456334|26.391237|-0.065097|9.699145|9.646425|0.994564|-0.115619|
|square_mse|raw|paired_real_ND_DEV|0|27.252453|26.973826|-0.278626|3.728773|3.707903|0.994403|-0.525844|
|square_mse|raw|paired_real_ND_DEV|1|27.004095|26.917720|-0.086376|3.576601|3.578219|1.000453|-1.050531|
|square_mse|recursive_self05|davis_train_pilot|0|26.800044|26.769294|-0.030749|6.545945|6.503381|0.993498|-0.058730|
|square_mse|recursive_self05|davis_train_pilot|1|26.612977|26.528624|-0.084352|6.393445|6.379258|0.997781|-0.133784|
|square_mse|recursive_self05|paired_real_ND_DEV|0|27.292047|27.028197|-0.263850|2.605355|2.596737|0.996692|-0.465043|
|square_mse|recursive_self05|paired_real_ND_DEV|1|26.923483|26.798087|-0.125396|2.534350|2.536276|1.000760|-1.119989|
|plain|raw|davis_train_pilot|0|26.089592|26.072830|-0.016761|9.419586|9.569473|1.015912|-0.099993|
|plain|raw|davis_train_pilot|1|26.386187|26.297105|-0.089082|9.602678|9.757695|1.016143|-0.296990|
|plain|raw|paired_real_ND_DEV|0|25.477019|24.487801|-0.989218|3.435879|3.444468|1.002500|-1.377558|
|plain|raw|paired_real_ND_DEV|1|29.388992|29.071413|-0.317579|3.567660|3.535093|0.990872|-0.680271|
|plain|recursive_self05|davis_train_pilot|0|26.250709|26.267683|0.016974|6.305463|6.363328|1.009177|-0.056428|
|plain|recursive_self05|davis_train_pilot|1|26.560911|26.532456|-0.028455|6.371495|6.426511|1.008635|-0.208244|
|plain|recursive_self05|paired_real_ND_DEV|0|25.410948|24.450897|-0.960051|2.501577|2.506322|1.001897|-1.242833|
|plain|recursive_self05|paired_real_ND_DEV|1|29.327666|29.007175|-0.320492|2.520404|2.507401|0.994841|-0.739787|
|joint|raw|davis_train_pilot|0|26.091586|26.063746|-0.027841|10.337596|10.459323|1.011775|-0.114054|
|joint|raw|davis_train_pilot|1|26.343268|26.230106|-0.113161|10.390793|10.560272|1.016310|-0.304878|
|joint|raw|paired_real_ND_DEV|0|25.497429|24.521728|-0.975701|3.664179|3.658678|0.998499|-1.389701|
|joint|raw|paired_real_ND_DEV|1|29.201878|28.948750|-0.253129|3.794335|3.766806|0.992745|-0.575039|
|joint|recursive_self05|davis_train_pilot|0|26.543494|26.539210|-0.004283|6.649947|6.700262|1.007566|-0.094124|
|joint|recursive_self05|davis_train_pilot|1|26.786250|26.735777|-0.050473|6.668539|6.730638|1.009312|-0.216495|
|joint|recursive_self05|paired_real_ND_DEV|0|25.467942|24.512687|-0.955255|2.579340|2.578828|0.999802|-1.272138|
|joint|recursive_self05|paired_real_ND_DEV|1|29.232020|28.971819|-0.260200|2.603779|2.591660|0.995346|-0.648474|

|模型|读出|域|原平均绝对seed差|校准后|比|
|---|---|---|---:|---:|---:|
|square_mse|raw|davis_train_pilot|0.196665|0.196726|1.000310|
|square_mse|raw|paired_real_ND_DEV|1.195441|1.825570|1.527110|
|square_mse|recursive_self05|davis_train_pilot|0.229755|0.253629|1.103910|
|square_mse|recursive_self05|paired_real_ND_DEV|1.182361|1.794935|1.518094|
|plain|raw|davis_train_pilot|0.317868|0.289437|0.910559|
|plain|raw|paired_real_ND_DEV|3.911974|4.583612|1.171688|
|plain|recursive_self05|davis_train_pilot|0.315863|0.296573|0.938930|
|plain|recursive_self05|paired_real_ND_DEV|3.916719|4.556278|1.163290|
|joint|raw|davis_train_pilot|0.294707|0.218770|0.742329|
|joint|raw|paired_real_ND_DEV|3.704449|4.427022|1.195055|
|joint|recursive_self05|davis_train_pilot|0.260347|0.228873|0.879108|
|joint|recursive_self05|paired_real_ND_DEV|3.764078|4.459132|1.184655|

```json
{
  "square_mse": {
    "raw_quality_time_guard": false,
    "ND_seed_gap_reduced_25pct": false,
    "domain_mean_gain_020dB": false,
    "raw_reference_eligible": false,
    "recursive_quality_time_guard": false,
    "recursive_reference_eligible": false
  },
  "plain": {
    "raw_quality_time_guard": false,
    "ND_seed_gap_reduced_25pct": false,
    "domain_mean_gain_020dB": false,
    "raw_reference_eligible": false,
    "recursive_quality_time_guard": false,
    "recursive_reference_eligible": false
  },
  "joint": {
    "raw_quality_time_guard": false,
    "ND_seed_gap_reduced_25pct": false,
    "domain_mean_gain_020dB": false,
    "raw_reference_eligible": false,
    "recursive_quality_time_guard": false,
    "recursive_reference_eligible": false
  }
}
```

只有BN缓冲区变化；不能从本控制推断所有seed差异均由BN造成。已知校准方法不建立论文创新/未见夜间泛化/手机时延。停止样本量、momentum、域配比和seed局部网格。
