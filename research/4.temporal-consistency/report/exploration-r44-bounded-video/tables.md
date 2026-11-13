# R44 fixed factorial: complete final evaluation

All four models and both seeds, inherited R41 4000 plus new4000 each, fresh Adam. 240 new quality rows / 80 CNN sequences. Both domains and every condition are retained; no seed/checkpoint selection.

|Domain|Model|Output|Parameters|PSNR|SSIM|Temporal↓|Edge↓|
|---|---|---|---:|---:|---:|---:|---:|
|davis_train_pilot|bounded_mse|finite_self05|4077|26.591350|0.754729|7.223247|3.921564|
|davis_train_pilot|bounded_mse|raw|4077|26.447809|0.716360|9.892480|4.269065|
|davis_train_pilot|bounded_mse|recursive_self05|4077|26.693887|0.767682|6.460988|3.816490|
|davis_train_pilot|bounded_temporal|finite_self05|4077|26.437502|0.754482|6.953240|3.943484|
|davis_train_pilot|bounded_temporal|raw|4077|26.428317|0.722787|9.315320|4.210444|
|davis_train_pilot|bounded_temporal|recursive_self05|4077|26.494711|0.765138|6.249731|3.860952|
|davis_train_pilot|square_mse|finite_self05|4075|26.604146|0.754513|7.232934|3.927123|
|davis_train_pilot|square_mse|raw|4075|26.465786|0.716221|9.900441|4.277081|
|davis_train_pilot|square_mse|recursive_self05|4075|26.706510|0.767504|6.469695|3.821135|
|davis_train_pilot|square_temporal|finite_self05|4075|26.467987|0.754548|6.962661|3.946030|
|davis_train_pilot|square_temporal|raw|4075|26.462726|0.722861|9.327973|4.214644|
|davis_train_pilot|square_temporal|recursive_self05|4075|26.526049|0.765249|6.257393|3.862918|
|paired_real_ND_DEV|bounded_mse|finite_self05|4077|27.519685|0.920442|2.816229|1.419152|
|paired_real_ND_DEV|bounded_mse|raw|4077|27.586391|0.912358|3.660172|1.519279|
|paired_real_ND_DEV|bounded_mse|recursive_self05|4077|27.544106|0.923630|2.566485|1.383202|
|paired_real_ND_DEV|bounded_temporal|finite_self05|4077|27.446487|0.920771|2.738068|1.423879|
|paired_real_ND_DEV|bounded_temporal|raw|4077|27.546673|0.914870|3.483971|1.500279|
|paired_real_ND_DEV|bounded_temporal|recursive_self05|4077|27.459133|0.923281|2.503695|1.395458|
|paired_real_ND_DEV|square_mse|finite_self05|4075|27.080975|0.918989|2.815893|1.419971|
|paired_real_ND_DEV|square_mse|raw|4075|27.128274|0.911261|3.652687|1.517297|
|paired_real_ND_DEV|square_mse|recursive_self05|4075|27.107765|0.922044|2.569853|1.385233|
|paired_real_ND_DEV|square_temporal|finite_self05|4075|27.002126|0.919315|2.739802|1.424316|
|paired_real_ND_DEV|square_temporal|raw|4075|27.083676|0.913687|3.481611|1.498605|
|paired_real_ND_DEV|square_temporal|recursive_self05|4075|27.018224|0.921723|2.508414|1.396856|

|Comparison|All domain/seed quality|All domain/seed time|Worst condition ΔPSNR|Joint screen|
|---|---|---|---:|---|
|activation_mse|True|False|-0.253362|False|
|objective_square|False|False|-0.382986|False|
|objective_bounded|False|False|-0.351882|False|
|structure|True|False|-0.235324|False|
|stateless_vs_mse_recursive|False|False|-1.308472|False|
|stateless_vs_temporal_recursive|False|False|-1.115486|False|
|combined_vs_mse_recursive|False|False|-0.520547|False|
|combined_vs_temporal_recursive|False|False|-0.230589|False|

Registered structure: False; stateless deployment: False; combination quality: False. Combination cost has not been measured; it cannot be reported as the stateless result.

Activation/individual-objective comparisons are descriptive controls. Registered gates require every domain/seed PSNR≥−.05dB, temporal ratio≤.90 and all20 condition PSNR deltas≥−.5dB. Rational operations remain in inference; finite/recursive use explicit own-raw motion/history. GT motion is training-only and estimated, not motion truth. No final-test, phone cost or novelty claim follows from these tables.
