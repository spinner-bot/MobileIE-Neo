# R74完整固定对照

6 SYN条件来自3视频，4 ND开发源；无独立最终或手机结果。空间固定8000，两个motion各新4000，不合计为空间12000。

|域|seed|模型|history|输出|总参数|PSNR|Etemp|JOD|
|---|---:|---|---|---|---:|---:|---:|---:|
|davis_train_pilot|0|R44_plain|current|raw|4075|26.475237|10.101737|7.040340|
|davis_train_pilot|0|R44_plain|current|recursive_self05|4075|26.800044|6.545945|7.423528|
|paired_real_ND_DEV|0|R44_plain|current|raw|4075|27.252453|3.728773|8.414068|
|paired_real_ND_DEV|0|R44_plain|current|recursive_self05|4075|27.292047|2.605355|8.373159|
|davis_train_pilot|1|R44_plain|current|raw|4075|26.456334|9.699145|7.110548|
|davis_train_pilot|1|R44_plain|current|recursive_self05|4075|26.612977|6.393445|7.408986|
|paired_real_ND_DEV|1|R44_plain|current|raw|4075|27.004095|3.576601|8.552201|
|paired_real_ND_DEV|1|R44_plain|current|recursive_self05|4075|26.923483|2.534350|8.512007|
|davis_train_pilot|0|task|causal|raw|25993|26.515803|9.772469|7.057206|
|davis_train_pilot|0|task|zero_flow|raw|25993|26.510161|9.772626|7.045270|
|davis_train_pilot|0|proxy|causal|raw|25993|23.346046|8.535960|6.250717|
|davis_train_pilot|0|proxy|zero_flow|raw|25993|23.326294|8.574036|6.234128|
|paired_real_ND_DEV|0|task|causal|raw|25993|27.218049|3.636952|8.413408|
|paired_real_ND_DEV|0|task|zero_flow|raw|25993|27.215593|3.636686|8.409115|
|paired_real_ND_DEV|0|proxy|causal|raw|25993|24.969266|3.353737|7.525462|
|paired_real_ND_DEV|0|proxy|zero_flow|raw|25993|24.966197|3.370873|7.520210|
|davis_train_pilot|1|task|causal|raw|25993|26.477611|9.407812|7.118643|
|davis_train_pilot|1|task|zero_flow|raw|25993|26.472624|9.405967|7.108683|
|davis_train_pilot|1|proxy|causal|raw|25993|23.500248|8.314803|6.364857|
|davis_train_pilot|1|proxy|zero_flow|raw|25993|23.490139|8.351730|6.359157|
|paired_real_ND_DEV|1|task|causal|raw|25993|27.002902|3.545716|8.551431|
|paired_real_ND_DEV|1|task|zero_flow|raw|25993|27.002170|3.545698|8.550460|
|paired_real_ND_DEV|1|proxy|causal|raw|25993|25.076913|3.226000|7.832660|
|paired_real_ND_DEV|1|proxy|zero_flow|raw|25993|25.052047|3.250713|7.813291|
