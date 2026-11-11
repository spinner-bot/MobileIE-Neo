# Frozen current-global-gate intervention

All four modes and seeds; fixed step4000, noreset/raw, matched CPU references.

| Variant | Path | PSNR | SSIM | Temporal | Edge |
|---|---|---:|---:|---:|---:|
|R20_postwarp01|reference|27.213708|0.920250|3.128185|1.451064|
|R20_postwarp05|reference|27.056327|0.921553|2.703859|1.422808|
|R20_raw|reference|27.169147|0.916230|3.453515|1.494217|
|post_fixedhalf|causal|27.055305|0.921478|2.811711|1.424614|
|post_fixedhalf|current|27.200801|0.911815|3.667242|1.547227|
|post_fixedhalf|current_gate|27.120633|0.921597|2.798669|1.424114|
|pre_fixedhalf|causal|25.983390|0.920317|2.783137|1.436683|
|pre_fixedhalf|current|26.182870|0.911140|3.554433|1.558890|
|pre_fixedhalf|current_gate|26.076430|0.920177|2.755769|1.440895|
|pre_learnhalf|causal|26.061267|0.922263|2.591937|1.379524|
|pre_learnhalf|current|26.128143|0.915783|3.196725|1.467590|
|pre_learnhalf|current_gate|26.073317|0.922318|2.566576|1.381807|
|pre_learnzero|causal|26.161492|0.923788|2.679689|1.393447|
|pre_learnzero|current|26.107872|0.915083|3.401961|1.499707|
|pre_learnzero|current_gate|26.090212|0.923852|2.651387|1.392490|
