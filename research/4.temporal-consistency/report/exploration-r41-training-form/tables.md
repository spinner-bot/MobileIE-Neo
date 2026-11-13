# R41 strict-FP32 training-form control

Corrected stage002, final4000 only, two training forms × two seeds. Both train under strict FP32 with the same windows, optimizer and update count. Expanded raw convolution executes the preserved outputs separately, without adding parameters; this numerical implementation differs from the original single wide-kernel call. Both deployments are the same 4075-parameter slim class.

The primary comparison is expanded versus this stage’s matched slim FP32. R40 and other legacy references remain separate: their training precision was not this matched control. Equal updates/images do not imply equal training FLOPs. No warm-up, official full-schedule reproduction, video innovation, phone cost or independent test is claimed.

## davis_train_pilot

|Anchor family|Output|PSNR|SSIM|Temporal↓|Edge↓|
|---|---|---:|---:|---:|---:|
|R34_original|finite_self05|26.465757|0.743040|7.542986|4.014079|
|R34_original|raw|26.234161|0.698872|10.505352|4.446445|
|R34_original|recursive_self05|26.610354|0.758387|6.734680|3.882416|
|R40_mixed|finite_self05|25.630993|0.727212|7.959654|4.102731|
|R40_mixed|raw|25.133622|0.671224|11.362815|4.686064|
|R40_mixed|recursive_self05|25.803268|0.745086|7.105806|3.937862|
|R41_expanded_fp32|finite_self05|26.394347|0.751125|7.285598|3.948008|
|R41_expanded_fp32|raw|26.185509|0.711587|10.007045|4.312240|
|R41_expanded_fp32|recursive_self05|26.502948|0.764433|6.520411|3.839379|
|R41_slim_fp32|finite_self05|25.627506|0.727188|7.959190|4.102653|
|R41_slim_fp32|raw|25.130525|0.671213|11.362138|4.685800|
|R41_slim_fp32|recursive_self05|25.799645|0.745059|7.105422|3.937824|

## paired_real_ND_DEV

|Anchor family|Output|PSNR|SSIM|Temporal↓|Edge↓|
|---|---|---:|---:|---:|---:|
|R26_plain_MSE|finite_self05|27.036311|0.919373|2.649882|1.423199|
|R26_plain_MSE|raw|27.138015|0.914749|3.312686|1.482404|
|R26_plain_MSE|recursive_self05|27.014906|0.921483|2.430531|1.399981|
|R34_original|finite_self05|23.417486|0.890365|2.597015|1.592994|
|R34_original|raw|23.453047|0.888458|3.165706|1.628944|
|R34_original|recursive_self05|23.379556|0.890675|2.429468|1.582884|
|R40_mixed|finite_self05|24.776650|0.912866|2.784050|1.447481|
|R40_mixed|raw|24.860774|0.902075|3.627781|1.575744|
|R40_mixed|recursive_self05|24.784900|0.916657|2.540240|1.402203|
|R41_expanded_fp32|finite_self05|25.734394|0.916057|2.803777|1.449319|
|R41_expanded_fp32|raw|25.846515|0.908162|3.629984|1.549785|
|R41_expanded_fp32|recursive_self05|25.758288|0.919261|2.553407|1.412203|
|R41_slim_fp32|finite_self05|24.776960|0.912889|2.783841|1.447414|
|R41_slim_fp32|raw|24.861383|0.902098|3.627413|1.575703|
|R41_slim_fp32|recursive_self05|24.785150|0.916680|2.540057|1.402136|
|official_LOL|finite_self05|18.466035|0.870795|3.163622|1.583560|
|official_LOL|raw|18.492855|0.847843|4.379450|1.797023|
|official_LOL|recursive_self05|18.464390|0.877091|2.873811|1.516453|

|Domain|Seed|Expanded ΔPSNR vs matched slim|Matched slim ΔPSNR vs old R40|
|---|---:|---:|---:|
|davis_train_pilot|0|0.967692|-0.000919|
|davis_train_pilot|1|1.142276|-0.005274|
|paired_real_ND_DEV|0|0.170846|0.003131|
|paired_real_ND_DEV|1|1.799418|-0.001913|

Registered same-domain, both-seed baseline screen: True; qualifying domains: ['davis_train_pilot']. Both seeds must gain .2dB in the same domain and lose no more than .05dB in the other. This is a baseline screen, not temporal-method validation.

All raw/finite/recursive results and pairs remain visible. Finite/recursive use own-raw half-FB and explicit RGB history, without extra teacher or GT reset. Four real sources are reused daytime ND DEV; synthetic conditions are three source videos, not six independent benchmarks. Initial numeric gates do not establish bitwise-equivalent training.
