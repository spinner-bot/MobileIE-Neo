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
|R28 post_transport/charb/current/raw|trained candidate, CUDA; registered screening failed|25.350512|0.908679|3.624398|1.530704|
|R28 post_transport/charb/current/postwarp05|same-weight reference, CUDA; classical post|25.201555|0.917476|2.773998|1.419358|
|R28 post_transport/charb/causal/raw|trained candidate, CUDA; registered screening failed|25.139877|0.917585|2.782038|1.419822|
|R28 post_transport/mse/current/raw|trained candidate, CUDA; registered screening failed|26.872779|0.910133|3.598826|1.532307|
|R28 post_transport/mse/current/postwarp05|same-weight reference, CUDA; classical post|26.805066|0.919185|2.760614|1.423614|
|R28 post_transport/mse/causal/raw|trained candidate, CUDA; registered screening failed|26.780321|0.919689|2.775676|1.418377|

| 方法 | 预测参数 | 共享质量评价teacher参数 | 预测Conv GMAC@1080p | 额外历史状态 | Flow | 未来帧 | 手机ms |
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
|R28 post_transport/charb/current/raw|4075|0|7.938|0|False|0|未测|
|R28 post_transport/charb/current/postwarp05|4075|4075|7.938|RGB history 23.73 MiB@1080p + motion state|True|0|未测|
|R28 post_transport/charb/causal/raw|4075|4075|7.938|12ch feature history 94.92 MiB@1080p + motion state|True|0|未测|
|R28 post_transport/mse/current/raw|4075|0|7.938|0|False|0|未测|
|R28 post_transport/mse/current/postwarp05|4075|4075|7.938|RGB history 23.73 MiB@1080p + motion state|True|0|未测|
|R28 post_transport/mse/causal/raw|4075|4075|7.938|12ch feature history 94.92 MiB@1080p + motion state|True|0|未测|

运动列对应当前共享流场质量评价的固定官方MobileIE + 半分辨率Farneback/FB，并非经典post所有部署都需要第二个CNN。自驱运动post可复用自身增强输出，见下面外部同底座表。预计算flow不等于部署免费；需要独立teacher时要计入其权重/前向、CPUflow和warp。MAC列只计预测Conv2d，不含FST/attention/激活/warp/运动teacher；历史状态是对应方法的最低历史张量，不含完整峰值内存。

R27 post_transport真实推理依赖过去特征与flow；其约27.06/2.71是CPU冻结探针，不是无flow的单帧结果，也不能由空间重参数化消除独立过去输入。单帧current是另一个部署映射，R28须单独评价。

R28的4416行共同CUDA评价已完成，current/raw与causal/raw均未通过注册质量—时序筛查。causal需要flow和12ch历史状态，主路线冻结；同权重current+post保留完整，不能把训练后空间映射变化归为真实历史收益。

外部开发评价：[R30 DAVIS](../../docs/exploration-r30-protocol.md)为3段官方TRAIN来源、6个合成低光条件、每段64帧；双seed312行[共同CPU结果](../exploration-r30/tables.md)已完成，仍未建立候选联合优势，与上表CUDA原生域分开。[R31真实手机](../../docs/exploration-r31-protocol.md)为3设备各64帧，无同步GT，[原亮度预览](../exploration-r31-data/phone_inputs.png)，尚无方法质量结论。

外部有效性缺口：完整标准公开视频验证、真实手机域方法评价、同域训练/适配的已发表对照、手机端完整延迟及独立保留测试仍待。FastDVDnet/StableLLVE已有迁移评价，尚不能替代这些缺口。

完整机器表：[evidence.csv](evidence.csv)；数据来源：[sources.json](sources.json)。

## 外部同底座CPU证据（独立表）

以下均为R26 plain MSE双seed、同一R30六条件。与上方native/CUDA表分开；仅自驱post已测CPU驻留输入流水线，不能当手机延迟。

|方法|PSNR|SSIM|时序误差↓|边缘误差↓|CPU流水线ms/帧@480×854|
|---|---:|---:|---:|---:|---:|
|fastdvd_sigma15_five_current|23.853137|0.713621|8.742266|4.268499|未测|
|fastdvd_sigma15_window|23.872768|0.718324|7.720899|4.174024|未测|
|raw|23.661207|0.678138|10.069569|4.437215|未测|
|self_motion_post05|23.817468|0.712322|7.448614|4.123568|200.612|
|shared_motion_post05|23.735146|0.707393|7.673995|4.158096|未测|

自驱post没有独立teacher前向，状态为一张raw增强RGB；该CPU循环总时间含输出留存指针开销，分帧计时不含后续指标/文件保存。未测完整峰值内存或相机到屏幕延迟。[外部机器表](external_evidence.csv)。

预训练来源限制：官方FastDVDnet文档训练清单覆盖R30三开发片及DAVIS TRAIN60/VAL30，因此新的R35迁移结果不能声称未见视频泛化；旧R12的ND原生表不因该DAVIS重叠自动变成训练片。已仅按元数据保留另30段test-dev来源/1788帧，与该90训练ID及项目训练/开发ID不重叠，JPEG成员尚未读取。这仍不是官方LLVE标准基准，其他公开权重来源需分别核验。

## R34公共动态训练后的固定双seed对照

以下三控制均同资料/同更新预算，采用最终4000权重。质量是R30六条件/三源视频的共同CPU均值；与上面R26旧训练资料/原生CUDA表分开。成本若已测，仅来自固定单片前16帧、两seed各三轮的CPU驻留输入流水线，不能当手机速度。

|模型|历史|输出|PSNR|SSIM|时序↓|边缘↓|CPU中位ms/帧|
|---|---|---|---:|---:|---:|---:|---:|
|causal|causal|raw|25.496515|0.696354|9.551315|4.374354|131.757|
|causal|current|postwarp05|25.358520|0.727633|7.361671|4.109950|未测|
|causal|current|raw|25.492768|0.696565|9.561273|4.373176|125.470|
|causal|current|self_motion_post05|25.587608|0.733710|7.052074|4.065406|237.708|
|original|current|postwarp05|26.218021|0.737103|7.833347|4.058940|未测|
|original|current|raw|26.234161|0.698872|10.505352|4.446445|94.528|
|original|current|self_motion_post05|26.465757|0.743040|7.542986|4.014079|190.884|
|spatial|current|postwarp05|25.341330|0.727188|7.310007|4.112746|未测|
|spatial|current|raw|25.490643|0.696783|9.461922|4.370254|131.030|
|spatial|current|self_motion_post05|25.570730|0.733230|7.002414|4.068684|235.643|

actual相对独立空间+self-post的注册联合门槛（双seed均过）：False；相对原始模型+self-post：False；相对自身current+self-post：False。门槛仅为DEV筛查，不能替代全部质量/成本、新颖性、手机和独立测试证据。

同权重current保留训练后的空间路径；actual内部仍有学习偏移/采样，不能称完全无对齐成本。shared-motion post仅用原共享teacher质量控制，未实测其完整部署成本。[动态机器表](dynamic_evidence.csv)、[配对与分seed全表](../exploration-r34/tables.md)。
