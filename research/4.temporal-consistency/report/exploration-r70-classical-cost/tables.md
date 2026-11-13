# R70 CPU2阶段描述性耗时

同一过滤结果跨seed复用，过滤仅计一次。共享服务器、未独立预热重复计时；不是手机/实时、完整FLOP或因果等待界。质量/IO未归入未明确包含的阶段，原完整过滤阶段含数组IO。

|域|mode|seed|尺寸|序列条件|阶段|秒/帧中位|最小|最大|
|---|---|---:|---|---:|---|---:|---:|---:|
|davis_train_pilot|spatial_only|None|480×854|6|native_filter|1.812497|1.804494|1.862951|
|davis_train_pilot|spatial_only|None|480×854|6|full_filter|1.823989|1.815641|1.906995|
|davis_train_pilot|video_default|None|480×854|6|native_filter|5.238138|5.208292|5.268876|
|davis_train_pilot|video_default|None|480×854|6|full_filter|5.249411|5.219705|5.280020|
|paired_real_ND_DEV|spatial_only|None|288×512|4|native_filter|0.673334|0.654663|0.691070|
|paired_real_ND_DEV|spatial_only|None|288×512|4|full_filter|0.676927|0.658293|0.694865|
|paired_real_ND_DEV|video_default|None|288×512|4|native_filter|1.770372|1.735405|1.772600|
|paired_real_ND_DEV|video_default|None|288×512|4|full_filter|1.774088|1.738965|1.775621|
|davis_train_pilot|spatial_only|0|480×854|6|neural|0.104279|0.097563|0.108010|
|davis_train_pilot|spatial_only|0|480×854|6|classic_post|0.102302|0.098963|0.107697|
|davis_train_pilot|spatial_only|1|480×854|6|neural|0.108159|0.087207|0.131295|
|davis_train_pilot|spatial_only|1|480×854|6|classic_post|0.102371|0.095169|0.108522|
|davis_train_pilot|video_default|0|480×854|6|neural|0.105068|0.086289|0.117193|
|davis_train_pilot|video_default|0|480×854|6|classic_post|0.097467|0.094926|0.102868|
|davis_train_pilot|video_default|1|480×854|6|neural|0.099557|0.088132|0.109827|
|davis_train_pilot|video_default|1|480×854|6|classic_post|0.097587|0.094526|0.101368|
|paired_real_ND_DEV|spatial_only|0|288×512|4|neural|0.031513|0.030884|0.032643|
|paired_real_ND_DEV|spatial_only|0|288×512|4|classic_post|0.032783|0.031710|0.034013|
|paired_real_ND_DEV|spatial_only|1|288×512|4|neural|0.031162|0.029981|0.031765|
|paired_real_ND_DEV|spatial_only|1|288×512|4|classic_post|0.032515|0.032418|0.033211|
|paired_real_ND_DEV|video_default|0|288×512|4|neural|0.031706|0.031499|0.032538|
|paired_real_ND_DEV|video_default|0|288×512|4|classic_post|0.032696|0.032350|0.033659|
|paired_real_ND_DEV|video_default|1|288×512|4|neural|0.032263|0.031489|0.034395|
|paired_real_ND_DEV|video_default|1|288×512|4|classic_post|0.032964|0.032496|0.033410|
