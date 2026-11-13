# R45 complete fixed controller comparison

Both seeds, three controls each4000 new updates, same frozen square-MSE base inherited8000. All alpha points retained; 180 quality rows, 20 new CNN sequences obtaining cues and replaying existing raw. DEV only.

|Domain|Method|Parameters|PSNR|SSIM|Temporal↓|Edge↓|
|---|---|---:|---:|---:|---:|---:|
|davis_train_pilot|fixed_01|4075|26.732729|0.732230|8.921811|4.115431|
|davis_train_pilot|fixed_025|4075|26.932460|0.751111|7.768194|3.940652|
|davis_train_pilot|fixed_05|4075|26.706510|0.767504|6.469695|3.821135|
|davis_train_pilot|fixed_075|4075|25.555811|0.751340|5.859992|4.011525|
|davis_train_pilot|fixed_09|4075|23.832652|0.693034|5.935660|4.519718|
|davis_train_pilot|raw|4075|26.465786|0.716221|9.900441|4.277081|
|davis_train_pilot|response|4116|27.115699|0.760573|7.121737|3.860483|
|davis_train_pilot|rgb|4116|27.112124|0.760378|7.140606|3.862030|
|davis_train_pilot|scalar|4076|26.921635|0.749133|7.892656|3.957834|
|paired_real_ND_DEV|fixed_01|4075|27.187630|0.916116|3.315930|1.465329|
|paired_real_ND_DEV|fixed_025|4075|27.217140|0.920867|2.940176|1.412129|
|paired_real_ND_DEV|fixed_05|4075|27.107765|0.922044|2.569853|1.385233|
|paired_real_ND_DEV|fixed_075|4075|26.720750|0.910647|2.475401|1.468985|
|paired_real_ND_DEV|fixed_09|4075|26.147374|0.886949|2.575231|1.662956|
|paired_real_ND_DEV|raw|4075|27.128274|0.911261|3.652687|1.517297|
|paired_real_ND_DEV|response|4116|27.231626|0.923067|2.741460|1.386956|
|paired_real_ND_DEV|rgb|4116|27.233645|0.923107|2.733771|1.386211|
|paired_real_ND_DEV|scalar|4076|27.217755|0.920442|2.979227|1.417152|

|Comparison|All domain/seed quality|All domain/seed time|Worst ΔPSNR|Gate|
|---|---|---|---:|---|
|response_vs_scalar|True|False|-0.037441|False|
|response_vs_rgb|True|False|-0.005907|False|
|response_vs_fixed05|True|False|-0.279218|False|
|rgb_vs_scalar|True|False|-0.035436|False|
|scalar_vs_fixed05|True|False|-0.534834|False|
|rgb_vs_fixed05|True|False|-0.279349|False|

Independent cue screen: False; fixed-alpha frontier nondominated: True; registered full quality screen: False. Cost not measured.

The response arm must pass scalar/RGB/fixed.5 comparisons jointly. Generic learned fusion or better constant alpha is not cue evidence. All stateful outputs retain own-raw motion and RGB history. Internal response cue is rescaled feature RMS, not calibrated uncertainty or an independent derivative invention.
