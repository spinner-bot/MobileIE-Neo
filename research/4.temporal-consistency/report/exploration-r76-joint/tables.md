# R76：更多计算的单观测自条件控制

self4每真实帧独立四次完整调用；causal每帧一次并读过去。参数相同，计算不相同。
原R75全部80条件与八门槛原样保留，新增20控制；仍为重复DEV，ND288×512原生裁块。

|域|seed|映射|PSNR|SSIM|Etemp↓|JOD↑|
|---|---:|---|---:|---:|---:|---:|
|davis_train_pilot|0|MobileIE_raw|26.475237|0.714517|10.101737|7.040340|
|davis_train_pilot|0|MobileIE_recursive_self05|26.800044|0.768807|6.545945|7.423528|
|davis_train_pilot|0|ReBotNet_causal|25.164926|0.640995|10.496574|6.989136|
|davis_train_pilot|0|ReBotNet_current|25.156037|0.640438|10.527278|7.024618|
|davis_train_pilot|0|ReBotNet_self4|25.176785|0.641561|10.542155|6.984848|
|davis_train_pilot|1|MobileIE_raw|26.456334|0.717926|9.699145|7.110548|
|davis_train_pilot|1|MobileIE_recursive_self05|26.612977|0.766202|6.393445|7.408986|
|davis_train_pilot|1|ReBotNet_causal|25.569037|0.636783|10.337987|7.286837|
|davis_train_pilot|1|ReBotNet_current|25.555152|0.636784|10.420951|7.295352|
|davis_train_pilot|1|ReBotNet_self4|25.567372|0.637191|10.418149|7.287075|
|paired_real_ND_DEV|0|MobileIE_raw|27.252453|0.909500|3.728773|8.414068|
|paired_real_ND_DEV|0|MobileIE_recursive_self05|27.292047|0.920910|2.605355|8.373159|
|paired_real_ND_DEV|0|ReBotNet_causal|32.626498|0.917034|3.201547|8.662256|
|paired_real_ND_DEV|0|ReBotNet_current|32.576200|0.916954|3.199230|8.661782|
|paired_real_ND_DEV|0|ReBotNet_self4|32.640556|0.917213|3.227214|8.665163|
|paired_real_ND_DEV|1|MobileIE_raw|27.004095|0.913022|3.576601|8.552201|
|paired_real_ND_DEV|1|MobileIE_recursive_self05|26.923483|0.923179|2.534350|8.512007|
|paired_real_ND_DEV|1|ReBotNet_causal|31.052930|0.915571|3.873731|8.485445|
|paired_real_ND_DEV|1|ReBotNet_current|31.236111|0.916072|3.980890|8.467765|
|paired_real_ND_DEV|1|ReBotNet_self4|31.079862|0.915648|4.471909|8.373874|

|域|seed|causal−self4 PSNR|Etemp比|JOD差|有限支持|
|---|---:|---:|---:|---:|---|
|davis_train_pilot|0|-0.011859|0.995676|+0.004288|False|
|davis_train_pilot|1|+0.001665|0.992306|-0.000238|False|
|paired_real_ND_DEV|0|-0.014058|0.992047|-0.002907|False|
|paired_real_ND_DEV|1|-0.026932|0.866236|+0.111571|True|

原R75八门槛全部通过=False；本控制四单元支持=False。
self4首帧四次、causal首帧一次，未剪掉首帧；20项首帧差另存。
若只赢raw-current而未赢self4，额外观测的独立作用保留未决；也不由失败否定所有时间依赖。
无新训练/独立视频/手机或独立最终；即使支持也不是新方法或一般因果证明。
