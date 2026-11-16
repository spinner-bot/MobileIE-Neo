# R80完整条件读出：启动阶段和固定尾段

原F0固定4075参数/8000更新；FST/RGB前缀拟合不使用GT，GT diagnostic只作不可部署诊断。full含启动；tail固定去前8帧，不同scope的JOD均独立计算。

|scope|域|seed|控制|PSNR|Etemp|JOD|
|---|---|---:|---|---:|---:|---:|
|full|davis_train_pilot|0|MobileIE_raw|26.475237|10.101737|7.040340|
|full|davis_train_pilot|0|MobileIE_recursive_self05|26.800044|6.545945|7.423528|
|full|paired_real_ND_DEV|0|MobileIE_raw|27.252453|3.728773|8.414068|
|full|paired_real_ND_DEV|0|MobileIE_recursive_self05|27.292047|2.605355|8.373159|
|full|davis_train_pilot|1|MobileIE_raw|26.456334|9.699145|7.110548|
|full|davis_train_pilot|1|MobileIE_recursive_self05|26.612977|6.393445|7.408986|
|full|paired_real_ND_DEV|1|MobileIE_raw|27.004095|3.576601|8.552201|
|full|paired_real_ND_DEV|1|MobileIE_recursive_self05|26.923483|2.534350|8.512007|
|full|davis_train_pilot|0|FST_prefix|26.285592|9.924575|7.066224|
|tail|davis_train_pilot|0|FST_prefix|26.279098|9.884492|7.030467|
|full|davis_train_pilot|0|RGB_prefix|26.419196|9.983806|7.060156|
|tail|davis_train_pilot|0|RGB_prefix|26.426232|9.947074|7.022928|
|full|davis_train_pilot|0|FST_GT_diagnostic|26.580636|9.944949|7.063027|
|tail|davis_train_pilot|0|FST_GT_diagnostic|26.588280|9.909356|7.026440|
|tail|davis_train_pilot|0|MobileIE_raw|26.484689|10.074248|6.999083|
|tail|davis_train_pilot|0|MobileIE_recursive_self05|26.809351|6.547799|7.474264|
|full|paired_real_ND_DEV|0|FST_prefix|27.254412|3.703198|8.408112|
|tail|paired_real_ND_DEV|0|FST_prefix|27.048796|3.743929|8.084598|
|full|paired_real_ND_DEV|0|RGB_prefix|27.278700|3.723911|8.412636|
|tail|paired_real_ND_DEV|0|RGB_prefix|27.079497|3.771072|8.097637|
|full|paired_real_ND_DEV|0|FST_GT_diagnostic|31.731881|3.998551|8.061475|
|tail|paired_real_ND_DEV|0|FST_GT_diagnostic|32.031976|3.775404|8.169997|
|tail|paired_real_ND_DEV|0|MobileIE_raw|27.046470|3.778966|8.102615|
|tail|paired_real_ND_DEV|0|MobileIE_recursive_self05|27.053917|2.668571|8.076820|
|full|davis_train_pilot|1|FST_prefix|26.128580|9.529005|7.121690|
|tail|davis_train_pilot|1|FST_prefix|26.104488|9.492126|7.084762|
|full|davis_train_pilot|1|RGB_prefix|26.300771|9.593110|7.120389|
|tail|davis_train_pilot|1|RGB_prefix|26.293531|9.558917|7.083328|
|full|davis_train_pilot|1|FST_GT_diagnostic|26.706481|9.699368|7.101100|
|tail|davis_train_pilot|1|FST_GT_diagnostic|26.716357|9.671412|7.062505|
|tail|davis_train_pilot|1|MobileIE_raw|26.461629|9.673866|7.071798|
|tail|davis_train_pilot|1|MobileIE_recursive_self05|26.619460|6.396739|7.442959|
|full|paired_real_ND_DEV|1|FST_prefix|27.016414|3.555583|8.547560|
|tail|paired_real_ND_DEV|1|FST_prefix|26.791233|3.588120|8.228760|
|full|paired_real_ND_DEV|1|RGB_prefix|27.028420|3.572631|8.550852|
|tail|paired_real_ND_DEV|1|RGB_prefix|26.806068|3.610793|8.239103|
|full|paired_real_ND_DEV|1|FST_GT_diagnostic|32.311751|3.904227|8.081663|
|tail|paired_real_ND_DEV|1|FST_GT_diagnostic|32.674912|3.617254|8.282624|
|tail|paired_real_ND_DEV|1|MobileIE_raw|26.773748|3.617264|8.244349|
|tail|paired_real_ND_DEV|1|MobileIE_recursive_self05|26.641353|2.597631|8.206411|
