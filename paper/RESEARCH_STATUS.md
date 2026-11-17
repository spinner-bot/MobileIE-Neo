# 论文证据与下一实验决策入口

2026-11-17T11:25:00.042085+08:00；Codex  (sqs_A100)。本页只整合已完成证据，不执行新CNN/SGD/DEV/GPU；R85原3200计时前向另有完整记录。这是研究决策板，不是投稿主表完成声明。内部PDF仍为已审阅build020。

## 当前可直接比较的开发表

以下逐项取自R82完整140共同条件/28域均值；两seed分别保留，不选择最好seed或checkpoint。SYN是三源六退化条件、64帧，ND为四实拍DEV源、16帧；均已反复使用，不能称独立最终或自然夜间。PSNR↑、Etemp↓、JOD↑；两域不平均成一个总分。JOD采用原固定名义30fps与显示条件，非MOS。训练窗R83/R84数值不加入该表。

|输出|seed|SYN PSNR|SYN Etemp|SYN JOD|ND PSNR|ND Etemp|ND JOD|
|---|---:|---:|---:|---:|---:|---:|---:|
|MobileIE raw|0|26.475237|10.101737|7.040340|27.252453|3.728773|8.414068|
|MobileIE raw|1|26.456334|9.699145|7.110548|27.004095|3.576601|8.552201|
|MobileIE + strong-rec|0|26.800044|6.545945|7.423528|27.292047|2.605355|8.373159|
|MobileIE + strong-rec|1|26.612977|6.393445|7.408986|26.923483|2.534350|8.512007|
|R82独立空间H|0|26.744387|9.233142|7.181342|26.894956|3.613104|8.385563|
|R82独立空间H|1|26.773071|9.300521|7.119835|27.557980|3.552874|8.553997|
|R82空间H + strong-rec|0|26.812507|6.225389|7.444736|26.907346|2.577920|8.345220|
|R82空间H + strong-rec|1|26.823626|6.243073|7.390307|27.370820|2.521956|8.510125|
|R82 direct|0|26.663762|9.636597|7.055743|26.900427|3.638654|8.375465|
|R82 direct|1|26.670762|9.508344|7.066219|27.568616|3.510716|8.563849|
|R82 centered|0|26.730275|8.813486|7.077999|27.241632|3.458869|8.398499|
|R82 centered|1|26.824044|8.469819|7.117442|27.337593|3.285188|8.539045|

R82 centered相对raw四组联合门槛均过，时序降7.24%–12.75%；但对F0 strong-rec时序仍高29.63%–34.64%，对独立空间H再strong-rec高30.26%–41.57%。新空间H也有明显逐源损失，不能仅看正面域均值。A/B各8/24比较通过，均未形成完整质量资格，完整成本资格False，方法/创新均未建立。[原完整表](../research/4.temporal-consistency/report/exploration-r82-joint/tables.md)、[门槛与逐源解释](../research/4.temporal-consistency/docs/exploration-r82-decision.md)。

## 效率列及不能混接的测量

|输出|注册参数|1080p Conv GMAC|完整计算流水线：常驻CPU/前16帧|手机时延|
|---|---:|---:|---|---|
|MobileIE raw|4075|7.938|R85已测，下面分域/seed报告|未测|
|MobileIE strong-rec|4075|7.938 + 未计运动/warp等|R85已测，下面分域/seed报告|未测|
|R82 spatial/direct/centered|4379|未测；centered调用H两次|按失败协议未扩成本|未测|

R85原唯一CPU1010313于11:20:34正常完成，3200真实骨干/1500 half-FB、80warmup/120正式/40等价资格，原输出与同权重前缀最大差0；无新SGD/GPU/质量指标。下表主量为每条件三次完整序列ms/frame的中位数，再在域内等条件平均；比值为匹配条件比值的平均，未必等于域均值相除。

|域/原尺寸|seed|raw ms/frame|strong-rec ms/frame|平均条件rec/raw比|
|---|---:|---:|---:|---:|
|SYN 480×854|0|107.218|204.910|1.915|
|SYN 480×854|1|112.490|208.734|1.856|
|ND 288×512|0|45.272|81.201|1.794|
|ND 288×512|1|44.662|82.614|1.850|

硬件Xeon Gold6248R，Torch2.9.1+cu126 intra2/inter1、OpenCV4.11.0 thread1、CPU FP32/B1，非独占。含输入/输出转换、真实CNN、原half-FB/warp/.5融合与输出sink拷贝；不含加载/解码、相机/文件/display I/O、sink分配、JOD或服务排队。原每模式2warmup+3正式全保留，不给三次样本虚假置信区间。质量表SYN用64帧，这里同源同权重成本用前16帧；不是逐帧完全对应的质量—成本配对，不能隐去startup占比和窗口范围。

strong-rec稳定持久RGB状态为SYN9838080B、ND3538944B；共用16帧宿主输出sink另外为78704640B/28311552B，不是模型历史状态。当前输入/flow/workspace/CNN激活不计入持久状态，不称峰值RAM或手机内存。[完整20条件/重复极值表](../research/4.temporal-consistency/report/exploration-r85-baseline-cost/tables.md)、[原结果](../research/4.temporal-consistency/process/exploration-r85/cpu001/summary.json)、[终态](../research/4.temporal-consistency/process/exploration-r85/terminal001.json)。

1.79–1.91倍仅属于当前half-FB参考的域-seed平均条件比，不能概括所有传统运动；[已核快速DIS参考缺口](../research/5.prior-art-survey/docs/69-fast-classical-flow-reference-gap.md)尚未有本项目质量/速度结果，不把其文献速度挪入本表。

Conv数字仅现有统一分辨率预算的卷积乘加，不计pool/activation/memory/运动等；不是端侧完整推理量。R34的CPU计时使用不同权重与限定16帧，不接到本表作为同模型手机速度。[原预算表](generated/reference-budget-table.tex)、[限定CPU参考](generated/classical-table.tex)。304注册H参数相同不等于严格相同有效自由度/优化过程，空间(c,c)输入有冗余方向，centered双H约束不同。

公开视频方法已有完成证据，不再把“尚未同表胜出”写成“完全没做”：[FastDVDnet及其它参考](generated/perceptual-reference-table.tex)、[StableLLVE任务适配](generated/stablellve-adapted-table.tex)、[ReBotNet-XS与self4](generated/causal-reference-table.tex)。它们的训练初始化、预算、未来输入/尺寸和公开权重来源限制各不相同：FastDVDnet 2479096参数、未来2帧；StableLLVE 4316259参数；ReBotNet-XS 890275参数、64K/384随机初始化训练。不能拿参数量代替速度，也不把未充分匹配的适配当作原论文极限。原R47重复源帧训练限制不得隐去，后续修订参考以各表来源为准。

## 收敛约束

- 已成立：问题与强后处理参考、双seed开发比较、公开方法适配及其限制均有证据；R82存在相对raw的收益，R83通过固定状态支持前提。
- 未成立：一个同时超过强rec、保留空间质量且具有可证新颖性的可部署方法。R84固定冻结特征硬匹配已失败，不追加半径/阈值/归一化网格。[R84决定](../research/4.temporal-consistency/docs/exploration-r84-decision.md)。
- 任何新正式候选须先说明不同于旧STOP的学习/信息路径及直接先例差异；使用同数据、独立空间训练参照及其strong-rec，不只对F0 raw。普通特征共享/相关/softmax/warp/KD不是贡献声明。
- 单帧零时域状态与有状态因果推理分两类报告；前者训练看视频不等于部署取得跨帧信息，后者参数少不等于零推理开销。新的正式预算与止损线须事前设定，不根据结果扩半径/seed或修改阈值。
- 自然动态配对、实际手机完整pipeline、创新差异与最终独立评估仍缺。未过开发资格的候选不扩大其外部/手机测试；保留最终像素继续封存。既有REDS外部开发、ND配对和无GT手机视频各有价值，但不能替代这些缺口。

[R28监督来源勘误](../research/4.temporal-consistency/docs/training-target-provenance-r83.md)说明旧R28没有strong-rec输出KD；这纠正“已经做过”的表述，不使普通输出蒸馏成为创新，也不构成自动重开原R28的理由。论文要从当前诊断型内部稿收敛到可投叙事，需解决上述方法缺口；本板不以整理证据代替这一目标。
