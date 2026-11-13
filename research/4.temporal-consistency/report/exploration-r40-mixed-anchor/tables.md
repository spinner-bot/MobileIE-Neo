# R40 fixed mixed-data anchor

Final4000 only, both seeds, common CPU. 60 new quality rows/20 new CNN sequences plus 96 complete same-CPU references. Domain means stay separate; no pooled score can hide a transfer loss. Original 4075-parameter backbone and optimization are retained. This is a strong-data baseline, not a new temporal method.

## davis_train_pilot

|Anchor family|Method|PSNR|SSIM|Temporal↓|Edge↓|
|---|---|---:|---:|---:|---:|
|R34_original|finite_self05|26.465757|0.743040|7.542986|4.014079|
|R34_original|raw|26.234161|0.698872|10.505352|4.446445|
|R34_original|recursive_self05|26.610354|0.758387|6.734680|3.882416|
|R40_mixed|finite_self05|25.630993|0.727212|7.959654|4.102731|
|R40_mixed|raw|25.133622|0.671224|11.362815|4.686064|
|R40_mixed|recursive_self05|25.803268|0.745086|7.105806|3.937862|

## paired_real_ND_DEV

|Anchor family|Method|PSNR|SSIM|Temporal↓|Edge↓|
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
|official_LOL|finite_self05|18.466035|0.870795|3.163622|1.583560|
|official_LOL|raw|18.492855|0.847843|4.379450|1.797023|
|official_LOL|recursive_self05|18.464390|0.877091|2.873811|1.516453|

|Seed|Synthetic ΔPSNR vs R34|Real ΔPSNR vs R34|Real ΔPSNR vs R26|Diagnostic gate|
|---|---:|---:|---:|---|
|0|-1.055170|1.061971|-1.657163|False|
|1|-1.145909|1.753482|-2.897319|False|

Raw diagnostic requires at least +2dB real recovery vs R34 and synthetic loss no greater than .2dB, both seeds. The stronger R26 real reference is retained explicitly. Passing only restores a baseline; it does not prove novel video modeling, actual phone quality/latency or independent generalization. Post uses own adjacent raw half-FB; recursive also retains previous fused output. No extra teacher or GT reset. Real sources are four reused daytime ND crops, not natural-night test videos.
