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

## R36强原始底座上的传统递归对照

同R34原始final4000权重与同R30输入，两seed完整48行。有限post只使用上一raw，递归EMA另保存上一融合输出；三post共享同一own-raw半尺度FB运动，部署不需要第二CNN。

|方法|PSNR|SSIM|时序↓|边缘↓|在线历史MiB@480×854|同轮CPU中位ms/帧|
|---|---:|---:|---:|---:|---:|---:|
|raw|26.234161|0.698872|10.505352|4.446445|0.000|100.890|
|finite_self05|26.465757|0.743040|7.542986|4.014079|4.691|184.124|
|recursive_self05|26.610354|0.758387|6.734680|3.882416|9.382|186.455|
|adaptive_recursive|26.385385|0.708503|9.945699|4.346142|10.946|未测|

固定递归EMA相对有限post双seed联合筛查：True；启发式adaptive：False。递归是既有传统机制，不算创新。后续模型必须同时对照这个更强参考；不能只打败有限两帧post。

这些为三开发源上的固定控制。自适应方差未经校准，不等价复现BNLK/EMVD；共享质量阶段计时不作独立流水线速度。若CPU栏已填，来自R36独立raw/finite/recursive同轮18次流水线，固定单片前16帧，六个run均值取中位；不能和上方R34旧轮次混排，min–max见机器表。手机仍未测。[传统机器表](classical_evidence.csv)、[逐seed与限制](../exploration-r36-classical/tables.md)。

## R37保留强空间映射的全分辨率状态：完整负结果

冻结同一R34原始骨干、仅训练512时间参数；final4000两个seed完整72行，含48行同anchor强传统参考复用。current显式返回不可变原骨干。

|方法|PSNR|SSIM|时序↓|边缘↓|在线历史MiB@480×854|
|---|---:|---:|---:|---:|---:|---:|
|raw|26.234161|0.698872|10.505352|4.446445|0.000|
|finite_self05|26.465757|0.743040|7.542986|4.014079|4.691|
|recursive_self05|26.610354|0.758387|6.734680|3.882416|9.382|
|adaptive_recursive|26.385385|0.708503|9.945699|4.346142|10.946|
|zero_alignment|26.254584|0.701275|10.048953|4.385512|2.346|
|learned_alignment|26.281465|0.702914|9.818070|4.339232|2.346|

两个时间控制对finite/recursive的双seed主要门槛全部False，已停止该设计的局部网格。缓存减少是真实取舍，不能充当质量联合胜出；未为失败候选追加手机/CPU成本调优。[机器表](preserved_evidence.csv)、[全条件配对](../exploration-r37/tables.md)、[止损决策](../../docs/exploration-r37-decision.md)。

## R39固定空间底座的既有配对实拍DEV转移

四个既有白天ND实拍native crop，各16帧；全部同CPU2，官方一份权重、R26/R34各两个final4000 seed。训练方案不同，不能归于单个数据/优化因素；与R30合成域分开，不是自然夜景或独立最终确认。

|底座|输出|PSNR|SSIM|时序↓|边缘↓|
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

[完整共同CPU转移表](../exploration-r39-anchor-transfer/tables.md)、[机器表](paired_real_anchor_transfer.csv)。R34合成DEV较强，配对实拍却明显弱于R26；后续必须处理跨域映射及数据假设，不能只优化合成均值。

## R38真实手机输入上的参考检查

三设备各64帧、官方及R34双seed底座的raw/finite/recursive共27输出序列已完成。固定0/32/63接触表与全部64帧预览均保留，出现明显截断/色偏；没有同步GT，不提供伪PSNR/SSIM或真实闪烁排名。它们是参考行为诊断，不是R37候选结果或手机时延。[固定预览与限制](../exploration-r38-phone/README.md)。

## R40固定50/50资料混合：两域分别评价

保持原4075参数骨干和R34优化预算，只替换每步一半窗口为既有12视频真实配对；两个seed最终4000、60新质量行及96同CPU参考。既有R26训练方案还含其它差异，不能把R26差值孤立归于资料。以下域均值分开，不计算掩盖一侧损失的混合总分。

|域|底座|输出|PSNR|SSIM|时序↓|边缘↓|
|---|---|---|---:|---:|---:|---:|
|davis_train_pilot|R34_original|finite_self05|26.465757|0.743040|7.542986|4.014079|
|davis_train_pilot|R34_original|raw|26.234161|0.698872|10.505352|4.446445|
|davis_train_pilot|R34_original|recursive_self05|26.610354|0.758387|6.734680|3.882416|
|davis_train_pilot|R40_mixed|finite_self05|25.630993|0.727212|7.959654|4.102731|
|davis_train_pilot|R40_mixed|raw|25.133622|0.671224|11.362815|4.686064|
|davis_train_pilot|R40_mixed|recursive_self05|25.803268|0.745086|7.105806|3.937862|
|paired_real_ND_DEV|R26_plain_MSE|finite_self05|27.036311|0.919373|2.649882|1.423199|
|paired_real_ND_DEV|R26_plain_MSE|raw|27.138015|0.914749|3.312686|1.482404|
|paired_real_ND_DEV|R26_plain_MSE|recursive_self05|27.014906|0.921483|2.430531|1.399981|
|paired_real_ND_DEV|R34_original|finite_self05|23.417486|0.890365|2.597015|1.592994|
|paired_real_ND_DEV|R34_original|raw|23.453047|0.888458|3.165706|1.628944|
|paired_real_ND_DEV|R34_original|recursive_self05|23.379556|0.890675|2.429468|1.582884|
|paired_real_ND_DEV|R40_mixed|finite_self05|24.776650|0.912866|2.784050|1.447481|
|paired_real_ND_DEV|R40_mixed|raw|24.860774|0.902075|3.627781|1.575744|
|paired_real_ND_DEV|R40_mixed|recursive_self05|24.784900|0.916657|2.540240|1.402203|
|paired_real_ND_DEV|official_LOL|finite_self05|18.466035|0.870795|3.163622|1.583560|
|paired_real_ND_DEV|official_LOL|raw|18.492855|0.847843|4.379450|1.797023|
|paired_real_ND_DEV|official_LOL|recursive_self05|18.464390|0.877091|2.873811|1.516453|

注册raw诊断门槛（两seed均须真实域相对R34补回2dB、合成域不低于−.2dB）：False。同时保留更强R26真实域参考，不把部分补回称追平。即使通过也仅形成更强数据基线，不自动成为方法、新颖性或独立最终验证。

没有新权重手机延迟或整条CPU速度结论；纯单帧预测/finite/recursive路径分开，后两者部署有motion与历史。实拍域是四段既有白天ND crop。[完整两域与配对](../exploration-r40-mixed-anchor/tables.md)、[机器表](mixed_anchor_evidence.csv)。

## R41同精度训练参数化强对照

修正stage002两训练形式×两seed各4000更新，最终eval/fold门槛通过后共同CPU评价120新行/40CNN序列。两形式都为严格FP32；展开形式训练含MBR/BN，部署折叠回同4075参数。相同窗口、优化器和更新数不代表训练FLOPs相同。

原宽卷积按预训练输出块分开执行以保持初始函数数值门槛，执行实现与上游不同；无warm-up/完整原训练复现。先前失败与1568部分更新另存，不计本阶段模型预算，不用部分权重挑结果。旧R40训练精度不同，仅作单列历史参考。

|域|训练形式|输出|PSNR|SSIM|时序↓|边缘↓|
|---|---|---|---:|---:|---:|---:|
|davis_train_pilot|R41_expanded_fp32|finite_self05|26.394347|0.751125|7.285598|3.948008|
|davis_train_pilot|R41_expanded_fp32|raw|26.185509|0.711587|10.007045|4.312240|
|davis_train_pilot|R41_expanded_fp32|recursive_self05|26.502948|0.764433|6.520411|3.839379|
|davis_train_pilot|R41_slim_fp32|finite_self05|25.627506|0.727188|7.959190|4.102653|
|davis_train_pilot|R41_slim_fp32|raw|25.130525|0.671213|11.362138|4.685800|
|davis_train_pilot|R41_slim_fp32|recursive_self05|25.799645|0.745059|7.105422|3.937824|
|paired_real_ND_DEV|R41_expanded_fp32|finite_self05|25.734394|0.916057|2.803777|1.449319|
|paired_real_ND_DEV|R41_expanded_fp32|raw|25.846515|0.908162|3.629984|1.549785|
|paired_real_ND_DEV|R41_expanded_fp32|recursive_self05|25.758288|0.919261|2.553407|1.412203|
|paired_real_ND_DEV|R41_slim_fp32|finite_self05|24.776960|0.912889|2.783841|1.447414|
|paired_real_ND_DEV|R41_slim_fp32|raw|24.861383|0.902098|3.627413|1.575703|
|paired_real_ND_DEV|R41_slim_fp32|recursive_self05|24.785150|0.916680|2.540057|1.402136|

展开相对本阶段同精度slim的注册强基线门槛：True。要求同一域两seed均≥+.2dB、另一域均≥−.05dB；这是基线筛查，不是时间机制、新颖性或独立泛化证据。

raw无历史或flow，finite/recursive有传统历史与motion开销；手机延迟未测。两域分开，不合并成功分数。[完整全表及逐seed配对](../exploration-r41-training-form/tables.md)、[机器表](training_form_evidence.csv)。

## R42固定已知TLC空间统计对照

同R41展开最终4000两个seed，只将GAP替为窗口192的精确TLC局部均值；权重不变、零新训练。60新行/20CNN序列与同权重global60行复用，raw/finite/recursive完整保留。不是新视频方法，raw仍无时间状态。

|域|空间统计|输出|PSNR|SSIM|时序↓|边缘↓|
|---|---|---|---:|---:|---:|---:|
|davis_train_pilot|R41_expanded_fp32|finite_self05|26.394347|0.751125|7.285598|3.948008|
|davis_train_pilot|R41_expanded_fp32|raw|26.185509|0.711587|10.007045|4.312240|
|davis_train_pilot|R41_expanded_fp32|recursive_self05|26.502948|0.764433|6.520411|3.839379|
|davis_train_pilot|R42_TLC192|finite_self05|26.100293|0.750149|7.335667|3.958061|
|davis_train_pilot|R42_TLC192|raw|25.825225|0.709727|10.105171|4.333441|
|davis_train_pilot|R42_TLC192|recursive_self05|26.218988|0.763867|6.559867|3.844880|
|paired_real_ND_DEV|R41_expanded_fp32|finite_self05|25.734394|0.916057|2.803777|1.449319|
|paired_real_ND_DEV|R41_expanded_fp32|raw|25.846515|0.908162|3.629984|1.549785|
|paired_real_ND_DEV|R41_expanded_fp32|recursive_self05|25.758288|0.919261|2.553407|1.412203|
|paired_real_ND_DEV|R42_TLC192|finite_self05|23.465614|0.911440|2.915793|1.453470|
|paired_real_ND_DEV|R42_TLC192|raw|23.519425|0.903328|3.754367|1.556719|
|paired_real_ND_DEV|R42_TLC192|recursive_self05|23.494847|0.914719|2.663861|1.415607|

固定TLC注册门槛：False；要求同域双seed≥+.1dB、另域均≥−.05dB且所有域/seed raw时间比≤1.02。最差逐条件raw ΔPSNR为-3.716017dB。

参数4075不变，但att卷积从全局一点扩至全分辨率，1080p约新增.299卷积GMAC，前缀和与内存另计；不是零推理开销，当前不据共享质量计时排行手机或独立CPU速度。[完整表](../exploration-r42-tlc/tables.md)、[机器表](tlc_evidence.csv)。
