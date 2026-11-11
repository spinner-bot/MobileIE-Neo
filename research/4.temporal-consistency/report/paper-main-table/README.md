# 论文主表工作稿：已有证据与缺口

这是旧探索DEV证据清单，尚非标准基准投稿主表。不能跨证据组把异域权重迁移或CPU微小差当作公平SOTA排名。所有手机延迟均未测。

| 方法 | 证据组 | PSNR | SSIM | 时序误差↓ | 边缘误差↓ |
|---|---|---:|---:|---:|---:|
|MobileIE plain/charb/raw|matched task, CUDA|27.173261|0.916237|3.453193|1.494143|
|MobileIE plain/charb/postwarp05|matched task, CUDA|27.060087|0.921555|2.703716|1.422791|
|MobileIE plain/mse/raw|matched task, CUDA|27.142768|0.914767|3.312517|1.482224|
|MobileIE plain/mse/postwarp05|matched task, CUDA|27.037014|0.919573|2.640266|1.420639|
|R26 pre/MSE backbone/current/raw|matched task, CUDA|27.448107|0.913931|3.473010|1.498423|
|R26 pre/MSE backbone/current/postwarp05|matched task, CUDA|27.382816|0.922495|2.701053|1.393296|
|R9 resize1000 + FastDVDnet sigma5|published-weight transfer, CUDA; different base training|25.240443|0.921150|3.338726|1.434131|
|R9 resize1000 + FastDVDnet sigma15|published-weight transfer, CUDA; different base training|25.220569|0.924037|2.584182|1.349588|
|R9 resize1000 + FastDVDnet sigma30|published-weight transfer, CUDA; different base training|24.737507|0.900474|2.384989|1.544741|
|StableLLVE official clean|published-weight transfer, CUDA; not same-domain retraining|20.588452|0.852861|3.126237|1.863627|
|R27 frozen/copy/postwarp05|frozen mechanism, CPU; compare only this CPU pair|27.056327|0.921553|2.703859|1.422808|
|R27 frozen/post_transport/raw|frozen mechanism, CPU; compare only this CPU pair|27.062302|0.921865|2.709848|1.422163|
|R28 post_transport/charb/current|candidate pending; single-frame deployment|待完成|待完成|待完成|待完成|
|R28 post_transport/charb/causal|candidate pending; temporal deployment|待完成|待完成|待完成|待完成|
|R28 post_transport/mse/current|candidate pending; single-frame deployment|待完成|待完成|待完成|待完成|
|R28 post_transport/mse/causal|candidate pending; temporal deployment|待完成|待完成|待完成|待完成|

| 方法 | 预测参数 | 运动teacher参数 | 预测Conv GMAC@1080p | 额外历史状态 | Flow | 未来帧 | 手机ms |
|---|---:|---:|---:|---|---|---:|---|
|MobileIE plain/charb/raw|4075|0|7.938|0|False|0|未测|
|MobileIE plain/charb/postwarp05|4075|4075|7.938|RGB history 23.73 MiB@1080p + motion state|True|0|未测|
|MobileIE plain/mse/raw|4075|0|7.938|0|False|0|未测|
|MobileIE plain/mse/postwarp05|4075|4075|7.938|RGB history 23.73 MiB@1080p + motion state|True|0|未测|
|R26 pre/MSE backbone/current/raw|4075|0|7.938|0|False|0|未测|
|R26 pre/MSE backbone/current/postwarp05|4075|4075|7.938|RGB history 23.73 MiB@1080p + motion state|True|0|未测|
|R9 resize1000 + FastDVDnet sigma5|2483171|0|未计|5-frame window; peak state unmeasured|False|2|未测|
|R9 resize1000 + FastDVDnet sigma15|2483171|0|未计|5-frame window; peak state unmeasured|False|2|未测|
|R9 resize1000 + FastDVDnet sigma30|2483171|0|未计|5-frame window; peak state unmeasured|False|2|未测|
|StableLLVE official clean|4316259|0|未计|0|False|0|未测|
|R27 frozen/copy/postwarp05|4075|4075|7.938|RGB history 23.73 MiB@1080p + motion state|True|0|未测|
|R27 frozen/post_transport/raw|4075|4075|7.938|12ch feature history 94.92 MiB@1080p + motion state|True|0|未测|
|R28 post_transport/charb/current|4075|0|7.938|0|False|0|未测|
|R28 post_transport/charb/causal|4075|4075|7.938|12ch feature history 94.92 MiB@1080p + motion state|True|0|未测|
|R28 post_transport/mse/current|4075|0|7.938|0|False|0|未测|
|R28 post_transport/mse/causal|4075|4075|7.938|12ch feature history 94.92 MiB@1080p + motion state|True|0|未测|

运动列对应当前共享流场评价的固定官方MobileIE + 半分辨率Farneback/FB。预计算flow不等于部署免费；需要独立teacher时要计入其权重/前向、CPUflow和warp。MAC列只计预测Conv2d，不含FST/attention/激活/warp/运动teacher；历史状态是对应方法的最低历史张量，不含完整峰值内存。

R27 post_transport真实推理依赖过去特征与flow；其约27.06/2.71是CPU冻结探针，不是无flow的单帧结果，也不能由空间重参数化消除独立过去输入。单帧current是另一个部署映射，R28须单独评价。

外部输入已准备：[R30 DAVIS](../../docs/exploration-r30-protocol.md)为3段官方TRAIN来源、6个合成低光条件、每段64帧；[R31真实手机](../../docs/exploration-r31-protocol.md)为3设备各64帧，无同步GT，[原亮度预览](../exploration-r31-data/phone_inputs.png)。两者均为开发资料，方法外部评价尚未完成，不将已取得输入写作基准结果。

外部有效性缺口：完整标准公开视频验证、真实手机域方法评价、同域训练/适配的已发表对照、手机端完整延迟及独立保留测试仍待。FastDVDnet/StableLLVE已有迁移评价，尚不能替代这些缺口。

完整机器表：[evidence.csv](evidence.csv)；数据来源：[sources.json](sources.json)。
