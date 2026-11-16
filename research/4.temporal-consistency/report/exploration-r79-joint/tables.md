# R79完整训练信息条件对照

重复DEV：六SYN条件来自三源、四ND源。所有新控制均24955参数/4000更新；既有MobileIE参考4075参数/8000更新，非公平架构或训练算力排名。

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
|davis_train_pilot|0|spatial_current|25.216098|9.268390|6.973359|
|davis_train_pilot|0|balanced_causal|26.065071|9.541011|7.030983|
|davis_train_pilot|0|balanced_current|26.063499|9.556901|7.031704|
|davis_train_pilot|0|curriculum_causal|25.323705|8.822004|6.915333|
|davis_train_pilot|0|curriculum_current|25.338364|8.893059|6.915485|
|paired_real_ND_DEV|0|spatial_current|26.977330|3.325231|8.456345|
|paired_real_ND_DEV|0|balanced_causal|26.042024|3.559952|8.408218|
|paired_real_ND_DEV|0|balanced_current|26.067210|3.549666|8.407064|
|paired_real_ND_DEV|0|curriculum_causal|27.426535|3.456882|7.986178|
|paired_real_ND_DEV|0|curriculum_current|27.233173|3.485905|7.972264|
|davis_train_pilot|1|spatial_current|26.080419|9.652905|6.989627|
|davis_train_pilot|1|balanced_causal|26.298788|9.537275|6.996596|
|davis_train_pilot|1|balanced_current|26.301482|9.542698|6.996807|
|davis_train_pilot|1|curriculum_causal|25.896703|9.701686|7.048816|
|davis_train_pilot|1|curriculum_current|25.885114|9.748800|7.044932|
|paired_real_ND_DEV|1|spatial_current|21.336166|3.508578|8.053969|
|paired_real_ND_DEV|1|balanced_causal|22.383735|3.392875|7.896712|
|paired_real_ND_DEV|1|balanced_current|22.402351|3.405748|7.895369|
|paired_real_ND_DEV|1|curriculum_causal|21.911031|3.489951|7.838625|
|paired_real_ND_DEV|1|curriculum_current|21.991194|3.506807|7.844651|
