# R75 公开因果XS：最终完整共同表

SYN六条件480×854/64帧，ND四条件288×512原生裁块/16帧；均重复DEV。
ReBotNet-XS随机初始化64K、384训练；MobileIE原8K/128训练；输入、像素、初始化和训练计算不同，不作单因果或充分原表复现。

|域|seed|映射|PSNR|SSIM|Etemp↓|edge↓|JOD↑|
|---|---:|---|---:|---:|---:|---:|---:|
|davis_train_pilot|0|MobileIE_raw|26.475237|0.714517|10.101737|4.304666|7.040340|
|davis_train_pilot|0|MobileIE_recursive_self05|26.800044|0.768807|6.545945|3.809294|7.423528|
|davis_train_pilot|0|ReBotNet_causal|25.164926|0.640995|10.496574|4.334495|6.989136|
|davis_train_pilot|0|ReBotNet_current|25.156037|0.640438|10.527278|4.336351|7.024618|
|davis_train_pilot|1|MobileIE_raw|26.456334|0.717926|9.699145|4.249495|7.110548|
|davis_train_pilot|1|MobileIE_recursive_self05|26.612977|0.766202|6.393445|3.832977|7.408986|
|davis_train_pilot|1|ReBotNet_causal|25.569037|0.636783|10.337987|4.345639|7.286837|
|davis_train_pilot|1|ReBotNet_current|25.555152|0.636784|10.420951|4.346983|7.295352|
|paired_real_ND_DEV|0|MobileIE_raw|27.252453|0.909500|3.728773|1.535234|8.414068|
|paired_real_ND_DEV|0|MobileIE_recursive_self05|27.292047|0.920910|2.605355|1.395794|8.373159|
|paired_real_ND_DEV|0|ReBotNet_causal|32.626498|0.917034|3.201547|1.520904|8.662256|
|paired_real_ND_DEV|0|ReBotNet_current|32.576200|0.916954|3.199230|1.520799|8.661782|
|paired_real_ND_DEV|1|MobileIE_raw|27.004095|0.913022|3.576601|1.499359|8.552201|
|paired_real_ND_DEV|1|MobileIE_recursive_self05|26.923483|0.923179|2.534350|1.374671|8.512007|
|paired_real_ND_DEV|1|ReBotNet_causal|31.052930|0.915571|3.873731|1.510068|8.485445|
|paired_real_ND_DEV|1|ReBotNet_current|31.236111|0.916072|3.980890|1.506534|8.467765|

|域|seed|参考|ΔPSNR|Etemp比|ΔJOD|联合门槛|
|---|---:|---|---:|---:|---:|---|
|davis_train_pilot|0|MobileIE_recursive_self05|-1.635118|1.603523|-0.434392|False|
|davis_train_pilot|0|ReBotNet_current|+0.008889|0.997083|-0.035482|False|
|davis_train_pilot|1|MobileIE_recursive_self05|-1.043940|1.616967|-0.122150|False|
|davis_train_pilot|1|ReBotNet_current|+0.013885|0.992039|-0.008515|False|
|paired_real_ND_DEV|0|MobileIE_recursive_self05|+5.334451|1.228833|+0.289097|False|
|paired_real_ND_DEV|0|ReBotNet_current|+0.050298|1.000724|+0.000474|False|
|paired_real_ND_DEV|1|MobileIE_recursive_self05|+4.129447|1.528491|-0.026562|False|
|paired_real_ND_DEV|1|ReBotNet_current|-0.183181|0.973082|+0.017680|False|

完整八门槛已评价=True；全部通过=False。
current为同权重历史干预，不是独立训练空间模型；低时序误差本身不证明保细节或新方法。
本表不覆盖自然夜间、整幅1080p、手机或独立最终；仅真实完整条件，未取最好seed/中间checkpoint。
