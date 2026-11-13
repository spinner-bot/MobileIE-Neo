# R64 完整dense骨干共同状态：最终双seed全表

原8000起点，三组各新增4000，两seed共24000；所有组相同clipped raw MSE/资料/Adam。
140新质量行/80CNN序列/3584中心/6272逐帧；SYN为3源各两个退化，ND四源，均重复使用DEV。
R44/R48/R45完整旧参考条件不同；旧R63人工压力资格仍失败，实际任务资格不能推广全输入。

|域|seed|映射|PSNR|SSIM|Etemp↓|edge↓|
|---|---:|---|---:|---:|---:|---:|
|davis_train_pilot|0|plain:raw|26.089592|0.722315|9.419586|4.222443|
|davis_train_pilot|0|plain:recursive_self05|26.250709|0.766516|6.305463|3.835053|
|davis_train_pilot|0|spatial:raw|26.107445|0.722795|9.319271|4.208095|
|davis_train_pilot|0|spatial:recursive_self05|26.240070|0.765773|6.263468|3.834098|
|davis_train_pilot|0|causal:raw|26.116064|0.722518|9.327150|4.209698|
|davis_train_pilot|0|causal:recursive_self05|26.241151|0.765560|6.271456|3.836404|
|davis_train_pilot|0|causal:current_raw|26.104739|0.722692|9.362531|4.209663|
|paired_real_ND_DEV|0|plain:raw|25.477019|0.911226|3.435879|1.544980|
|paired_real_ND_DEV|0|plain:recursive_self05|25.410948|0.918406|2.501577|1.442187|
|paired_real_ND_DEV|0|spatial:raw|25.552456|0.911805|3.428619|1.538363|
|paired_real_ND_DEV|0|spatial:recursive_self05|25.480533|0.918943|2.499712|1.436452|
|paired_real_ND_DEV|0|causal:raw|25.560551|0.911792|3.420883|1.538186|
|paired_real_ND_DEV|0|causal:recursive_self05|25.487784|0.918874|2.496209|1.437202|
|paired_real_ND_DEV|0|causal:current_raw|25.552798|0.911714|3.433496|1.538323|
|davis_train_pilot|0|R44_square:raw|26.475237|0.714517|10.101737|4.304666|
|davis_train_pilot|0|R44_square:finite_self05|26.674556|0.755036|7.330425|3.924356|
|davis_train_pilot|0|R44_square:recursive_self05|26.800044|0.768807|6.545945|3.809294|
|paired_real_ND_DEV|0|R44_square:raw|27.252453|0.909500|3.728773|1.535234|
|paired_real_ND_DEV|0|R44_square:finite_self05|27.260667|0.917721|2.853690|1.432206|
|paired_real_ND_DEV|0|R44_square:recursive_self05|27.292047|0.920910|2.605355|1.395794|
|davis_train_pilot|0|R48_plain:raw|26.089592|0.722315|9.419586|4.222443|
|davis_train_pilot|0|R48_plain:finite_self05|26.178282|0.755242|7.015638|3.925769|
|davis_train_pilot|0|R48_plain:recursive_self05|26.250709|0.766516|6.305463|3.835053|
|paired_real_ND_DEV|0|R48_plain:raw|25.477019|0.911226|3.435879|1.544980|
|paired_real_ND_DEV|0|R48_plain:finite_self05|25.401953|0.916247|2.721851|1.469690|
|paired_real_ND_DEV|0|R48_plain:recursive_self05|25.410948|0.918406|2.501577|1.442187|
|davis_train_pilot|0|R45_rgb|27.172215|0.759827|7.305598|3.867806|
|paired_real_ND_DEV|0|R45_rgb|27.387785|0.921679|2.790698|1.400289|
|davis_train_pilot|1|plain:raw|26.386187|0.720654|9.602678|4.222580|
|davis_train_pilot|1|plain:recursive_self05|26.560911|0.766316|6.371495|3.846999|
|davis_train_pilot|1|spatial:raw|26.405642|0.721853|9.463605|4.207429|
|davis_train_pilot|1|spatial:recursive_self05|26.539934|0.765941|6.314014|3.845317|
|davis_train_pilot|1|causal:raw|26.414864|0.721511|9.470229|4.210046|
|davis_train_pilot|1|causal:recursive_self05|26.538341|0.765690|6.322900|3.848846|
|davis_train_pilot|1|causal:current_raw|26.400866|0.721652|9.511755|4.209237|
|paired_real_ND_DEV|1|plain:raw|29.388992|0.919658|3.567660|1.479507|
|paired_real_ND_DEV|1|plain:recursive_self05|29.327666|0.929704|2.520404|1.357629|
|paired_real_ND_DEV|1|spatial:raw|29.456302|0.920407|3.535772|1.472108|
|paired_real_ND_DEV|1|spatial:recursive_self05|29.390781|0.930090|2.510524|1.353146|
|paired_real_ND_DEV|1|causal:raw|29.462500|0.920394|3.522903|1.471015|
|paired_real_ND_DEV|1|causal:recursive_self05|29.380355|0.929959|2.507199|1.354332|
|paired_real_ND_DEV|1|causal:current_raw|29.406164|0.920079|3.556887|1.472893|
|davis_train_pilot|1|R44_square:raw|26.456334|0.717926|9.699145|4.249495|
|davis_train_pilot|1|R44_square:finite_self05|26.533735|0.753989|7.135443|3.929889|
|davis_train_pilot|1|R44_square:recursive_self05|26.612977|0.766202|6.393445|3.832977|
|paired_real_ND_DEV|1|R44_square:raw|27.004095|0.913022|3.576601|1.499359|
|paired_real_ND_DEV|1|R44_square:finite_self05|26.901284|0.920257|2.778097|1.407736|
|paired_real_ND_DEV|1|R44_square:recursive_self05|26.923483|0.923179|2.534350|1.374671|
|davis_train_pilot|1|R48_plain:raw|26.386187|0.720654|9.602678|4.222580|
|davis_train_pilot|1|R48_plain:finite_self05|26.472375|0.754673|7.100426|3.935638|
|davis_train_pilot|1|R48_plain:recursive_self05|26.560911|0.766316|6.371495|3.846999|
|paired_real_ND_DEV|1|R48_plain:raw|29.388992|0.919658|3.567660|1.479507|
|paired_real_ND_DEV|1|R48_plain:finite_self05|29.325133|0.926632|2.774748|1.391541|
|paired_real_ND_DEV|1|R48_plain:recursive_self05|29.327666|0.929704|2.520404|1.357629|
|davis_train_pilot|1|R45_rgb|27.052034|0.760930|6.975614|3.856254|
|paired_real_ND_DEV|1|R45_rgb|27.079505|0.924536|2.676844|1.372133|

|域|seed|强post参考|causal raw ΔPSNR|Etemp比|最差条件ΔPSNR|
|---|---:|---|---:|---:|---:|
|davis_train_pilot|0|plain:recursive_self05|-0.134645|1.479217|-1.028654|
|davis_train_pilot|0|spatial:recursive_self05|-0.124006|1.489135|-1.005708|
|davis_train_pilot|0|R44_square:recursive_self05|-0.683980|1.424874|-1.466423|
|davis_train_pilot|1|plain:recursive_self05|-0.146047|1.486343|-1.121503|
|davis_train_pilot|1|spatial:recursive_self05|-0.125070|1.499875|-1.116615|
|davis_train_pilot|1|R44_square:recursive_self05|-0.198113|1.481240|-1.231547|
|paired_real_ND_DEV|0|plain:recursive_self05|+0.149604|1.367490|-0.025315|
|paired_real_ND_DEV|0|spatial:recursive_self05|+0.080018|1.368511|-0.190830|
|paired_real_ND_DEV|0|R44_square:recursive_self05|-1.731496|1.313020|-6.912523|
|paired_real_ND_DEV|1|plain:recursive_self05|+0.134833|1.397753|-0.201703|
|paired_real_ND_DEV|1|spatial:recursive_self05|+0.071719|1.403254|-0.548790|
|paired_real_ND_DEV|1|R44_square:recursive_self05|+2.539016|1.390062|-0.532924|

同权重历史贡献门槛=False；A质量资格=False；B质量资格=False。
B只容许保时间的资源取舍，仍须实际完整CPU≥20%节省；A仍须实际成本≤参考1.05。
这些门槛只针对causal raw，causal recursive不替代无外部flow问题；任何资格不代表创新/手机/最终验证。
