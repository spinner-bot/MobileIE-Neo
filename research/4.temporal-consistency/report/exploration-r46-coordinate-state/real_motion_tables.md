# R46实际运动完整结果

两个seed均正常结束，60新质量行/2688逐帧行/0CNN；ordinary重算核对R45 fixed .5。

|域|seed|方法|PSNR|SSIM|时间↓|edge↓|
|---|---:|---|---:|---:|---:|---:|
|davis_train_pilot|0|raw|26.475237|0.714517|10.101737|4.304666|
|davis_train_pilot|0|fixed_025|26.994866|0.750926|7.895790|3.944723|
|davis_train_pilot|0|ordinary|26.800044|0.768807|6.545945|3.809294|
|davis_train_pilot|0|same_mask|26.981744|0.761670|6.764128|3.864177|
|davis_train_pilot|0|anchor8|26.950815|0.760713|6.774210|3.876596|
|davis_train_pilot|0|fixed_075|25.644874|0.753813|5.901586|3.993384|
|davis_train_pilot|0|rgb|27.172215|0.759827|7.305598|3.867806|
|davis_train_pilot|1|raw|26.456334|0.717926|9.699145|4.249495|
|davis_train_pilot|1|fixed_025|26.870053|0.751296|7.640599|3.936582|
|davis_train_pilot|1|ordinary|26.612977|0.766202|6.393445|3.832977|
|davis_train_pilot|1|same_mask|26.824335|0.760029|6.576988|3.877588|
|davis_train_pilot|1|anchor8|26.794333|0.759017|6.585989|3.890022|
|davis_train_pilot|1|fixed_075|25.466748|0.748867|5.818399|4.029666|
|davis_train_pilot|1|rgb|27.052034|0.760930|6.975614|3.856254|
|paired_real_ND_DEV|0|raw|27.252453|0.909500|3.728773|1.535234|
|paired_real_ND_DEV|0|fixed_025|27.367955|0.919433|2.993846|1.426200|
|paired_real_ND_DEV|0|ordinary|27.292047|0.920910|2.605355|1.395794|
|paired_real_ND_DEV|0|same_mask|27.294921|0.920595|2.676645|1.403768|
|paired_real_ND_DEV|0|anchor8|27.281801|0.920206|2.680678|1.408969|
|paired_real_ND_DEV|0|fixed_075|26.931821|0.909790|2.496994|1.476427|
|paired_real_ND_DEV|0|rgb|27.387785|0.921679|2.790698|1.400289|
|paired_real_ND_DEV|1|raw|27.004095|0.913022|3.576601|1.499359|
|paired_real_ND_DEV|1|fixed_025|27.066325|0.922301|2.886506|1.398058|
|paired_real_ND_DEV|1|ordinary|26.923483|0.923179|2.534350|1.374671|
|paired_real_ND_DEV|1|same_mask|26.945994|0.922978|2.595866|1.380286|
|paired_real_ND_DEV|1|anchor8|26.935964|0.922563|2.599220|1.385843|
|paired_real_ND_DEV|1|fixed_075|26.509678|0.911504|2.453809|1.461544|
|paired_real_ND_DEV|1|rgb|27.079505|0.924536|2.676844|1.372133|

raw/fixed/RGB复用R45，其它三方法为本轮共同评价；全部相同R44最终空间底座。same_mask为额外拒绝匹配的诊断，不是免坐标成本的新方法。

|域|seed|主对照|ΔPSNR|时间比|最差条件ΔPSNR|画质/时间/条件通过|
|---|---:|---|---:|---:|---:|---|
|davis_train_pilot|0|ordinary|+0.150772|1.034871|-0.128013|True/False/True|
|davis_train_pilot|0|same_mask|-0.030928|1.001490|-0.057771|True/False/True|
|davis_train_pilot|1|ordinary|+0.181356|1.030116|-0.097417|True/False/True|
|davis_train_pilot|1|same_mask|-0.030003|1.001368|-0.053327|True/False/True|
|paired_real_ND_DEV|0|ordinary|-0.010246|1.028911|-0.110444|True/False/True|
|paired_real_ND_DEV|0|same_mask|-0.013120|1.001507|-0.035942|True/False/True|
|paired_real_ND_DEV|1|ordinary|+0.012481|1.025596|-0.069815|True/False/True|
|paired_real_ND_DEV|1|same_mask|-0.010030|1.001292|-0.022630|True/False/True|

完整质量门槛：False。候选持久历史比原ordinary多75%；无手机/零开销/新颖性或独立最终结论。实际motion不是GT，GT流仅用于既有评价。
