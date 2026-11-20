# T1同F0已知蒸馏对照完整开发表

重复DEV；普通KD对照，不主张方法创新；自然动态、手机和独立最终仍待证。

|域|seed|输出|PSNR|SSIM|Etemp|JOD|
|---|---:|---|---:|---:|---:|---:|
|davis_train_pilot|0|DIS_recursive_self05|27.050821|0.766023|6.520286|7.447420|
|davis_train_pilot|0|MobileIE_raw|26.475237|0.714517|10.101737|7.040340|
|davis_train_pilot|0|MobileIE_recursive_self05|26.800044|0.768807|6.545945|7.423528|
|davis_train_pilot|0|t1_gt_DIS05|26.944571|0.763223|6.418177|7.428125|
|davis_train_pilot|0|t1_gt_FB05|26.694278|0.765775|6.456677|7.403001|
|davis_train_pilot|0|t1_gt_raw|26.455992|0.716327|9.825273|7.056645|
|davis_train_pilot|0|t1_spatial_DIS05|26.978763|0.764663|6.448487|7.438185|
|davis_train_pilot|0|t1_spatial_FB05|26.727324|0.767265|6.484074|7.413366|
|davis_train_pilot|0|t1_spatial_raw|26.465784|0.716195|9.908504|7.045671|
|davis_train_pilot|0|t1_temporal_DIS05|26.888750|0.763330|6.349456|7.422710|
|davis_train_pilot|0|t1_temporal_FB05|26.633269|0.765436|6.397729|7.393900|
|davis_train_pilot|0|t1_temporal_raw|26.447809|0.718337|9.673088|7.056985|
|davis_train_pilot|1|DIS_recursive_self05|26.882562|0.763883|6.349339|7.440672|
|davis_train_pilot|1|MobileIE_raw|26.456334|0.717926|9.699145|7.110548|
|davis_train_pilot|1|MobileIE_recursive_self05|26.612977|0.766202|6.393445|7.408986|
|davis_train_pilot|1|t1_gt_DIS05|26.983573|0.764837|6.379454|7.453559|
|davis_train_pilot|1|t1_gt_FB05|26.722077|0.767108|6.420932|7.425241|
|davis_train_pilot|1|t1_gt_raw|26.526686|0.719024|9.757946|7.085891|
|davis_train_pilot|1|t1_spatial_DIS05|26.956892|0.763986|6.330856|7.443064|
|davis_train_pilot|1|t1_spatial_FB05|26.689627|0.766187|6.377763|7.412351|
|davis_train_pilot|1|t1_spatial_raw|26.541106|0.719281|9.635639|7.104335|
|davis_train_pilot|1|t1_temporal_DIS05|26.859641|0.762249|6.248235|7.420103|
|davis_train_pilot|1|t1_temporal_FB05|26.590599|0.764166|6.301906|7.385714|
|davis_train_pilot|1|t1_temporal_raw|26.508989|0.720824|9.427034|7.105532|
|paired_real_ND_DEV|0|DIS_recursive_self05|27.336567|0.917612|2.821074|8.400475|
|paired_real_ND_DEV|0|MobileIE_raw|27.252453|0.909500|3.728773|8.414068|
|paired_real_ND_DEV|0|MobileIE_recursive_self05|27.292047|0.920910|2.605355|8.373159|
|paired_real_ND_DEV|0|t1_gt_DIS05|26.207676|0.919114|2.775937|8.366849|
|paired_real_ND_DEV|0|t1_gt_FB05|26.161973|0.921858|2.583034|8.339309|
|paired_real_ND_DEV|0|t1_gt_raw|26.170173|0.911059|3.689219|8.373732|
|paired_real_ND_DEV|0|t1_spatial_DIS05|26.439054|0.918804|2.777948|8.381122|
|paired_real_ND_DEV|0|t1_spatial_FB05|26.393402|0.921489|2.588523|8.352991|
|paired_real_ND_DEV|0|t1_spatial_raw|26.394985|0.910590|3.694612|8.390727|
|paired_real_ND_DEV|0|t1_temporal_DIS05|26.409299|0.918890|2.746958|8.383237|
|paired_real_ND_DEV|0|t1_temporal_FB05|26.360900|0.921228|2.569185|8.355044|
|paired_real_ND_DEV|0|t1_temporal_raw|26.377257|0.911375|3.636666|8.395849|
|paired_real_ND_DEV|1|DIS_recursive_self05|27.001759|0.920617|2.719274|8.535978|
|paired_real_ND_DEV|1|MobileIE_raw|27.004095|0.913022|3.576601|8.552201|
|paired_real_ND_DEV|1|MobileIE_recursive_self05|26.923483|0.923179|2.534350|8.512007|
|paired_real_ND_DEV|1|t1_gt_DIS05|29.547418|0.924610|2.787274|8.584305|
|paired_real_ND_DEV|1|t1_gt_FB05|29.440834|0.928135|2.567936|8.568772|
|paired_real_ND_DEV|1|t1_gt_raw|29.436063|0.915570|3.715997|8.572973|
|paired_real_ND_DEV|1|t1_spatial_DIS05|28.662500|0.923821|2.729409|8.597972|
|paired_real_ND_DEV|1|t1_spatial_FB05|28.541189|0.926733|2.534459|8.578853|
|paired_real_ND_DEV|1|t1_spatial_raw|28.671280|0.915938|3.601176|8.601181|
|paired_real_ND_DEV|1|t1_temporal_DIS05|28.658957|0.923661|2.705561|8.596133|
|paired_real_ND_DEV|1|t1_temporal_FB05|28.533768|0.926508|2.516287|8.576074|
|paired_real_ND_DEV|1|t1_temporal_raw|28.683535|0.916696|3.547565|8.602217|

原F0恢复保护 3/4；主要A比较 0/40，B比较 0/40。

完整A质量资格=False；B质量资格=False。只有任一True才进入预定完整成本；这是质量资格，不是方法或速度成功。
