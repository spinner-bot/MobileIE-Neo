# R82固定共同条件修正完整表

重复DEV：六SYN条件来自三源、四ND源。F0固定4075参数/8000原更新，三H各304参数/8000新更新，合4379参数；centered双H，非零额外推理开销。错误历史仅外源teacher-forced诊断。

|域|seed|控制|PSNR|Etemp|JOD|
|---|---:|---|---:|---:|---:|
|davis_train_pilot|0|MobileIE_raw|26.475237|10.101737|7.040340|
|davis_train_pilot|0|MobileIE_recursive_self05|26.800044|6.545945|7.423528|
|paired_real_ND_DEV|0|MobileIE_raw|27.252453|3.728773|8.414068|
|paired_real_ND_DEV|0|MobileIE_recursive_self05|27.292047|2.605355|8.373159|
|davis_train_pilot|1|MobileIE_raw|26.456334|9.699145|7.110548|
|davis_train_pilot|1|MobileIE_recursive_self05|26.612977|6.393445|7.408986|
|paired_real_ND_DEV|1|MobileIE_raw|27.004095|3.576601|8.552201|
|paired_real_ND_DEV|1|MobileIE_recursive_self05|26.923483|2.534350|8.512007|
|davis_train_pilot|0|spatial|26.744387|9.233142|7.181342|
|davis_train_pilot|0|spatial_recursive_self05|26.812507|6.225389|7.444736|
|davis_train_pilot|0|direct|26.663762|9.636597|7.055743|
|davis_train_pilot|0|centered|26.730275|8.813486|7.077999|
|paired_real_ND_DEV|0|spatial|26.894956|3.613104|8.385563|
|paired_real_ND_DEV|0|spatial_recursive_self05|26.907346|2.577920|8.345220|
|paired_real_ND_DEV|0|direct|26.900427|3.638654|8.375465|
|paired_real_ND_DEV|0|centered|27.241632|3.458869|8.398499|
|davis_train_pilot|0|centered_wrong|26.139402|9.157219|7.080043|
|paired_real_ND_DEV|0|centered_wrong|27.303656|3.723328|8.334616|
|davis_train_pilot|1|spatial|26.773071|9.300521|7.119835|
|davis_train_pilot|1|spatial_recursive_self05|26.823626|6.243073|7.390307|
|davis_train_pilot|1|direct|26.670762|9.508344|7.066219|
|davis_train_pilot|1|centered|26.824044|8.469819|7.117442|
|paired_real_ND_DEV|1|spatial|27.557980|3.552874|8.553997|
|paired_real_ND_DEV|1|spatial_recursive_self05|27.370820|2.521956|8.510125|
|paired_real_ND_DEV|1|direct|27.568616|3.510716|8.563849|
|paired_real_ND_DEV|1|centered|27.337593|3.285188|8.539045|
|davis_train_pilot|1|centered_wrong|26.525830|8.899004|7.130672|
|paired_real_ND_DEV|1|centered_wrong|27.355562|3.703181|8.494243|
