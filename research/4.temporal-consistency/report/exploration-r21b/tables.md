# Aggregation-trained static information controls

All fixed endpoints; mean draw PSNR is averaged per draw, not PSNR of the mean output.

| Method | Path | rho | JPEG | Mean draw PSNR | Bias MSE | Draw variance |
|---|---|---:|---|---:|---:|---:|
|R14_plain|current|0|False|24.680145|0.00239151|0.00130036|
|R14_plain|current|0|True|24.237742|0.00237360|0.00168370|
|R14_plain|current|0.9|False|24.680145|0.00239151|0.00130036|
|R14_plain|current|0.9|True|24.237742|0.00237360|0.00168370|
|R14_plain|postaverage_clipped|0|False|25.579534|0.00237785|0.00064898|
|R14_plain|postaverage_clipped|0|True|25.330805|0.00236096|0.00084005|
|R14_plain|postaverage_clipped|0.9|False|24.778268|0.00239016|0.00121907|
|R14_plain|postaverage_clipped|0.9|True|24.492195|0.00237136|0.00146905|
|R14_plain|postaverage_thenclip|0|False|25.579534|0.00237785|0.00064898|
|R14_plain|postaverage_thenclip|0|True|25.330771|0.00236097|0.00084005|
|R14_plain|postaverage_thenclip|0.9|False|24.778268|0.00239016|0.00121907|
|R14_plain|postaverage_thenclip|0.9|True|24.492173|0.00237137|0.00146905|
|R14_plain|preaverage_current|0|False|25.672103|0.00224303|0.00068427|
|R14_plain|preaverage_current|0|True|25.102812|0.00243343|0.00085336|
|R14_plain|preaverage_current|0.9|False|24.777298|0.00237123|0.00123661|
|R14_plain|preaverage_current|0.9|True|24.448830|0.00236920|0.00147882|
|post_fixedhalf|current|0|False|24.584848|0.00183152|0.00199584|
|post_fixedhalf|current|0|True|24.037318|0.00182532|0.00246346|
|post_fixedhalf|current|0.9|False|24.584848|0.00183152|0.00199584|
|post_fixedhalf|current|0.9|True|24.037318|0.00182532|0.00246346|
|post_fixedhalf|learned_history|0|False|25.900243|0.00181317|0.00099675|
|post_fixedhalf|learned_history|0|True|25.532662|0.00180637|0.00122881|
|post_fixedhalf|learned_history|0.9|False|24.722798|0.00182931|0.00187059|
|post_fixedhalf|learned_history|0.9|True|24.388215|0.00181915|0.00214099|
|pre_fixedhalf|current|0|False|24.628193|0.00175168|0.00206530|
|pre_fixedhalf|current|0|True|23.911104|0.00179658|0.00261230|
|pre_fixedhalf|current|0.9|False|24.628193|0.00175168|0.00206530|
|pre_fixedhalf|current|0.9|True|23.911104|0.00179658|0.00261230|
|pre_fixedhalf|learned_history|0|False|26.257259|0.00156925|0.00097777|
|pre_fixedhalf|learned_history|0|True|25.507741|0.00176458|0.00120325|
|pre_fixedhalf|learned_history|0.9|False|24.796689|0.00172645|0.00192405|
|pre_fixedhalf|learned_history|0.9|True|24.324230|0.00178062|0.00218681|
|pre_learnhalf|current|0|False|24.123948|0.00280869|0.00130012|
|pre_learnhalf|current|0|True|23.866811|0.00253714|0.00182964|
|pre_learnhalf|current|0.9|False|24.123948|0.00280869|0.00130012|
|pre_learnhalf|current|0.9|True|23.866811|0.00253714|0.00182964|
|pre_learnhalf|learned_history|0|False|25.117631|0.00254404|0.00071026|
|pre_learnhalf|learned_history|0|True|24.947290|0.00243011|0.00093670|
|pre_learnhalf|learned_history|0.9|False|24.207862|0.00278462|0.00123726|
|pre_learnhalf|learned_history|0.9|True|24.139589|0.00251632|0.00157057|
|pre_learnzero|current|0|False|24.127581|0.00261258|0.00155498|
|pre_learnzero|current|0|True|23.721214|0.00253001|0.00198043|
|pre_learnzero|current|0.9|False|24.127581|0.00261258|0.00155498|
|pre_learnzero|current|0.9|True|23.721214|0.00253001|0.00198043|
|pre_learnzero|learned_history|0|False|25.218213|0.00238577|0.00079265|
|pre_learnzero|learned_history|0|True|24.952643|0.00239116|0.00098984|
|pre_learnzero|learned_history|0.9|False|24.239789|0.00257457|0.00146997|
|pre_learnzero|learned_history|0.9|True|24.014465|0.00250572|0.00169618|
