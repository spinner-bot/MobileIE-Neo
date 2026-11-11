# Matched-acceptance flow sensitivity

Common masks and reference flow are GT-assisted diagnostics, not deployment candidates.

| Variant | Regime | PSNR | SSIM | Temporal | Edge |
|---|---|---:|---:|---:|---:|
|R14_postwarp05|current|27.169147|0.916230|3.453515|1.494217|
|R14_postwarp05|reference_common|27.099427|0.923044|2.610968|1.408252|
|R14_postwarp05|teacher_common|27.082744|0.921962|2.693197|1.420430|
|R14_postwarp05|teacher_default|27.056327|0.921553|2.703859|1.422808|
|post_fixedhalf|current|27.200801|0.911815|3.667242|1.547227|
|post_fixedhalf|reference_common|27.102828|0.923147|2.704682|1.409054|
|post_fixedhalf|teacher_common|27.081184|0.922027|2.798390|1.421769|
|post_fixedhalf|teacher_default|27.055305|0.921478|2.811711|1.424614|
|pre_fixedhalf|current|26.182870|0.911140|3.554433|1.558890|
|pre_fixedhalf|reference_common|26.027320|0.921914|2.680564|1.421733|
|pre_fixedhalf|teacher_common|26.009501|0.920865|2.769136|1.433836|
|pre_fixedhalf|teacher_default|25.983390|0.920317|2.783137|1.436683|
|pre_learnhalf|current|26.128143|0.915783|3.196725|1.467590|
|pre_learnhalf|reference_common|26.080637|0.923425|2.520507|1.368201|
|pre_learnhalf|teacher_common|26.075107|0.922598|2.583157|1.377753|
|pre_learnhalf|teacher_default|26.061267|0.922263|2.591937|1.379524|
|pre_learnzero|current|26.107872|0.915083|3.401961|1.499707|
|pre_learnzero|reference_common|26.178377|0.924765|2.598843|1.383172|
|pre_learnzero|teacher_common|26.173156|0.924020|2.670736|1.392445|
|pre_learnzero|teacher_default|26.161492|0.923788|2.679689|1.393447|
