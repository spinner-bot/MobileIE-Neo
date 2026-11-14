# R30 external development results: completed phases only

Included phases: references, post-seed0, post-seed1. All rows use the common CPU backend. Each seed row averages six conditions from three source videos; these are not six independent videos.

Missing phases have no inferred values. Official pretrained MobileIE has no project training seed. Full registered model set is complete.

|Mode|Loss|Seed|History|Reset|Output|PSNR|SSIM|Temporal|Edge|
|---|---|---:|---|---|---|---:|---:|---:|---:|
|copy|charb|0|current|False|postwarp05|23.766534|0.707401|8.275877|4.184740|
|copy|charb|0|current|False|raw|23.652805|0.674183|11.097321|4.515418|
|copy|charb|0|current|True|postwarp05|23.686962|0.695573|9.248064|4.328700|
|copy|charb|0|current|True|raw|23.652805|0.674183|11.097321|4.515418|
|copy|charb|1|current|False|postwarp05|23.362694|0.701504|7.784162|4.164452|
|copy|charb|1|current|False|raw|23.271209|0.674364|10.283639|4.402127|
|copy|charb|1|current|True|postwarp05|23.291902|0.691583|8.701288|4.269787|
|copy|charb|1|current|True|raw|23.271209|0.674364|10.283639|4.402127|
|copy|mse|0|current|False|postwarp05|24.200568|0.713095|7.884127|4.128194|
|copy|mse|0|current|False|raw|24.120196|0.681078|10.429372|4.447802|
|copy|mse|0|current|True|postwarp05|24.123231|0.701606|8.776216|4.267146|
|copy|mse|0|current|True|raw|24.120196|0.681078|10.429372|4.447802|
|copy|mse|1|current|False|postwarp05|23.269723|0.701690|7.463862|4.187999|
|copy|mse|1|current|False|raw|23.202219|0.675199|9.709766|4.426628|
|copy|mse|1|current|True|postwarp05|23.214000|0.692129|8.257246|4.285103|
|copy|mse|1|current|True|raw|23.202219|0.675199|9.709766|4.426628|
|official|official|-1|current|False|postwarp05|17.897706|0.583748|10.171098|4.660538|
|official|official|-1|current|False|raw|17.251758|0.501455|15.475979|5.821514|
|official|official|-1|current|True|postwarp05|17.660819|0.558228|12.021067|5.096754|
|official|official|-1|current|True|raw|17.251758|0.501455|15.475979|5.821514|
|post_transport|charb|0|causal|False|raw|23.418897|0.700484|8.497494|4.109630|
|post_transport|charb|0|causal|True|raw|23.275627|0.685123|9.675613|4.316024|
|post_transport|charb|0|current|False|postwarp05|23.456537|0.701208|8.453884|4.100961|
|post_transport|charb|0|current|False|raw|23.165630|0.655911|11.741177|4.591076|
|post_transport|charb|0|current|True|postwarp05|23.311353|0.685609|9.635740|4.311207|
|post_transport|charb|0|current|True|raw|23.165630|0.655911|11.741177|4.591076|
|post_transport|charb|1|causal|False|raw|24.112892|0.696512|9.000059|4.160735|
|post_transport|charb|1|causal|True|raw|23.906035|0.680811|10.225098|4.351533|
|post_transport|charb|1|current|False|postwarp05|24.145209|0.697729|8.932069|4.147016|
|post_transport|charb|1|current|False|raw|23.742140|0.652471|12.461403|4.627285|
|post_transport|charb|1|current|True|postwarp05|23.930184|0.681641|10.170742|4.342753|
|post_transport|charb|1|current|True|raw|23.742140|0.652471|12.461403|4.627285|
|post_transport|mse|0|causal|False|raw|24.341150|0.707625|8.818913|4.110369|
|post_transport|mse|0|causal|True|raw|24.158319|0.691356|9.996475|4.324358|
|post_transport|mse|0|current|False|postwarp05|24.355389|0.708164|8.747284|4.101704|
|post_transport|mse|0|current|False|raw|24.002358|0.660825|12.142575|4.628111|
|post_transport|mse|0|current|True|postwarp05|24.178915|0.691785|9.910691|4.319132|
|post_transport|mse|0|current|True|raw|24.002358|0.660825|12.142575|4.628111|
|post_transport|mse|1|causal|False|raw|23.377027|0.706462|8.154554|4.124329|
|post_transport|mse|1|causal|True|raw|23.244389|0.691719|9.214264|4.298120|
|post_transport|mse|1|current|False|postwarp05|23.377799|0.706571|8.134822|4.121046|
|post_transport|mse|1|current|False|raw|23.119452|0.664785|11.109038|4.558600|
|post_transport|mse|1|current|True|postwarp05|23.243366|0.691760|9.196399|4.296201|
|post_transport|mse|1|current|True|raw|23.119452|0.664785|11.109038|4.558600|
|r26_pre_mse_spatial|mse|0|current|False|postwarp05|23.667837|0.682911|9.128427|4.200300|
|r26_pre_mse_spatial|mse|0|current|False|raw|23.207743|0.631261|13.039091|4.769716|
|r26_pre_mse_spatial|mse|0|current|True|postwarp05|23.462898|0.664795|10.518542|4.423986|
|r26_pre_mse_spatial|mse|0|current|True|raw|23.207743|0.631261|13.039091|4.769716|
|r26_pre_mse_spatial|mse|1|current|False|postwarp05|23.242079|0.676947|8.713836|4.195664|
|r26_pre_mse_spatial|mse|1|current|False|raw|22.912713|0.632595|12.217264|4.624689|
|r26_pre_mse_spatial|mse|1|current|True|postwarp05|23.101104|0.661729|9.933885|4.355107|
|r26_pre_mse_spatial|mse|1|current|True|raw|22.912713|0.632595|12.217264|4.624689|

Current/raw is single-frame. Current/post .5 and causal mappings need flow/history. This custom low-light DEV task is not an official DAVIS LLVE benchmark.
