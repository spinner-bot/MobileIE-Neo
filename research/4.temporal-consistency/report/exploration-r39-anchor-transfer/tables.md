# Fixed anchor transfer on four paired real ND DEV crops

Common CPU2/OpenCV1, FP32 B1, same four already-used native crops/GT/correspondence. Official weights have one fixed instance; R26/R34 families each have two final4000 seeds. The data/optimization/batch differ, so the family difference is not an isolated architectural effect. No independent final test, natural-night benchmark or phone latency.

|Anchor family|Method|PSNR|SSIM|Temporal↓|Edge↓|
|---|---|---:|---:|---:|---:|
|R26_plain_MSE|finite_self05|27.036311|0.919373|2.649882|1.423199|
|R26_plain_MSE|raw|27.138015|0.914749|3.312686|1.482404|
|R26_plain_MSE|recursive_self05|27.014906|0.921483|2.430531|1.399981|
|R34_original|finite_self05|23.417486|0.890365|2.597015|1.592994|
|R34_original|raw|23.453047|0.888458|3.165706|1.628944|
|R34_original|recursive_self05|23.379556|0.890675|2.429468|1.582884|
|official_LOL|finite_self05|18.466035|0.870795|3.163622|1.583560|
|official_LOL|raw|18.492855|0.847843|4.379450|1.797023|
|official_LOL|recursive_self05|18.464390|0.877091|2.873811|1.516453|

All post routes estimate half-FB from their own adjacent raw; finite uses previous raw, recursive uses previous fused. No extra teacher or GT reset. Old CUDA numbers are not inserted into this CPU table. All scenes and both trained seeds remain in the paired JSON.
