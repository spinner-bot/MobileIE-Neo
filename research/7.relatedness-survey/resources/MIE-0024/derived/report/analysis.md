# MIE-0024｜QATIE：量化部署的直接损失先例，尚无视频稳定性证据

2026-11-21；Codex  (sqs_A100) <langxibielangle@qq.com>。本资源为有效序27的论文项，同家族代码为MIE-0025；两项不能算两份独立实验。证据快照截止2026-11-17，正式版补取日期2026-11-21。

QATIE 对二期研究的主要价值，是明确要求同时核对训练、低精度工件和真实部署协议。其论文直接借鉴 MobileIE 的损失，骨干则来自另一路门控编码与多尺度精炼。作者报告 QAT 缓解 INT8 质量下降，但其表内仍有明显质量损失，且没有视频时序实验。应将它作为低精度对照与协议参考；不能以量化或门控本身作为二期创新依据。

## 身份、范围和原始来源

正式版为 [CVPR2026 Workshops/MAI，3847–3856](https://openaccess.thecvf.com/content/CVPR2026W/MAI/html/Thanh_Bridging_the_Training-Deployment_Gap_Gated_Encoding_and_Multi-Scale_Refinement_for_CVPRW_2026_paper.html)，对应 [arXiv 2604.21743v1](https://arxiv.org/abs/2604.21743v1)。预印本十页全文含36条参考已读；与正式版完整差异核查、关键3图5表实际查看，未发现科学文本/表值变化。作者链接的 [GenAI4E/QATIE](https://github.com/GenAI4E/QATIE/tree/6703a6d123042601a465378a74bdfe224696fe7d) 固定提交全部37工件/26适用文本及许可、两图、三权重已留存并静态分析。原件/获取记录见[来源清单](../../sources/manifest.json)，详细范围见[阅读笔记](../docs/reading-notes.md)、[版本与可得性](../docs/version-and-availability-review.json)、[实际视觉审阅](../docs/visual-review.json)。

MobileIE 的准确语境为正式p5、p7损失及p10[35]，不是骨干继承；p3–4所述DaHua-IIG先例[16]是已入目录MIE-0033，未为参考书目无限增加资源。没有把CVPR Workshop称主会，没有从GitHub owner推断作者身份，也未改变旧ID/顺序。

## 方法、信息条件与当前实现

输入是一张归一化RGB手机照片，监督是配对Canon照片。模型在S/2、S/4、S/8空间尺度做门控与UNet式精炼，decoder拼接上采样、精炼encoder及原分支skip，残差输出clamp到[0,1]。无历史/未来帧、跨输出state、运动对齐、时序loss或闪烁指标；零未来并不意味着已经解决逐帧闪烁。

当前代码35个卷积和5个InstanceNorm的字面求和为 P(c)=889c²+156c+3，c16/24/32/64分别230083/515811/915331/3651331，符合论文参数表的四舍五入。c32在1080×1920仅卷积约144.156672G MAC，另有norm、tanh、乘法、cat/搬运及量化成本。S/2的4c逻辑concat有66355200元素，不能把文件小当作低中间内存，更不能把该张量大小当实测峰值。详见[逐层字面算术](../docs/static-numeric-review.json)；没有导入作者模型或独立运行计数器。

论文推荐loss组合在源中为Y域PSNR、全图RGB余弦及相对离差加权L1。outlier权重有界且detach，提升相对大离差的权重；统一常偏差时该项可为零，不应泛化为整个目标无梯度。命名MSSSIM的类只计算单层SSIM，不能据此认定复现了表4多尺度对照；ImageNet VGG19仅用于另一loss3，不是论文推荐loss2的默认教师。

默认/演示CLI不构成论文原训练记录：c24/loss1、limit=.1，README另示c32/loss2、limit=.01；QAT又强制FP32。训练iPhone、test作val；独立glob缺逐文件配对核对，需在实验前固定。np.float32→ToTensor再/255正确。我们未取得数据像素或执行训练，因此没有声称实际错配、运行失败或用默认配置推翻论文。

## 训练到部署的证据差异

FX QAT路径使用qnnpack，head与默认ConcatFP32保留浮点，可从checkpoint读observer；另一路先移除observer给裸浮点权重，经过ONNX/onnx2tf，再做TF代表集PTQ。后一路默认取128个测试输入作校准，再在测试集评价，未隔离校准和最终评价；这不等于使用测试GT训练。历史表中究竟对应哪个版本/命令/文件，当前材料不足以确认。

两份公开TFLite不是仅尺寸不同，静态结构如下。数据来自[官方schema元数据审计](../docs/tflite-container-review001.json)，没有运行解释器。

| 工件 | 声明输入 → 输出 | 张量与算子 | 解释边界 |
|---|---|---|---|
| 100×100，1005112B | FLOAT32 NHWC [1,100,100,3] → FLOAT32 NCHW [1,3,100,100] | 185 INT8 +42 INT32 +2 FLOAT32；139op，浮点I/O有量化/反量化边界 | 内部INT8不等于整数I/O |
| 文件名1920×1080，1002624B | INT8 [1,1920,1080,3] → 同形状INT8 | 176 INT8 +40 INT32；128op，无float typedop；scale≈1/255、zero−128 | 实际H1920/W1080；不能改写为横屏形状 |

论文描述UINT8活动量，发布大文件signed INT8；仿射映射可能表达同范围，不能仅据符号判错。静态图类型也不证明QNN委派、文件与表身份、数值等价或手机速度。CKPT CRC与操作码可见c32/observer等文字，不能证明最优epoch、完整训练配置或原论文实测。

动态导出wrapper用reshape对齐输入形状；若原输出NCHW，则reshape不等于NHWC转置，这是需验证的条件风险。分块推理的线性mask外端点为零，在overlap>0的强制tiles/回退边界可能产生零权重黑边；原图直接匹配的路径不能据此一并判坏。InstanceNorm按当前图像/块空间统计，也会使分块与全图不必等价。上述均为静态推断，没有产生任何模型输出。

## 作者结果及其适用条件

| 作者实验 | 原文结果 | 对结论的约束 |
|---|---|---|
| 结构消融 | 完整22.194/.796/469ms；去res22.029/.793/464；去fusion21.940/.793/396；去resolution20.398/.789/237 | 提高质量同时显著增加成本，且改变容量 |
| 宽度消融 | c24 22.035/.785/249ms，c32 22.194/.796/469ms；c24作者score最高 | c32为质量取舍，不是所有目标均最优 |
| loss消融 | 推荐1.67h、22.194/.796；对照8.58h、19.79/.8763 | 对照SSIM更高，不能说全面占优；当前MS-SSIM实现身份待定 |
| INT8 PTQ → QAT | 20.576/.6139 → 21.050/.725 | +.474dB/+.1111SSIM，仍较FP32 QAT22.194/.796少1.144dB/.071 |
| HTP / GPU时延 | HTP FP32列151ms、INT8 QAT41.8ms；GPU469→319ms | HTP约3.61倍，41.8ms倒数约23.92FPS；不是30FPS/端到端持续视频 |

表5表头FP32与正文FP16描述不一致，保留原文而不替作者修订。INT8 PTQ HTP41.4ms略低于QAT41.8，不能把QAT说成比PTQ快。比赛表1用1024×1024、Capybara21.82/.7653、MOS3.2、GPU291/Arm266ms；与patch质量表及FullHD成本表不同，不能拼成同条件比较。

当前batch导出生成的部分CSV `device=cuda` 行仍调用CPU TFLite，源文档也说明只是标签/行数；它不是手机GPU或QNN基准。Torch benchmark另有实际设备路径，但默认1088×1920随机输入、20warmup/100runs/5trials，亦不等论文AI Benchmark。TFLite/PyTorch/训练的PSNR阈值及SSIM参数入口不同；原图配置与注释尺寸也不同。后续复现须统一数据配对、resize/shape、度量、delegate与计时范围，不能从文件名推断这些条件。

## 二期研究价值、反证与下一阶段交接

可迁移的优先事项是部署协议：固定同一权重/转换版本/输入范围/布局、校准集和实际设备图，保留浮点、PTQ、QAT三者质量差异及完整成本。QAT、门控编码、多尺度精炼、MobileIE损失已有明确先例；简单加入这些组件不足以支撑二期创新。

尚缺原始数据ID/许可和对齐、训练seed/命令、表与权重/手机工件对应SHA、逐图/逐trial记录、真实delegate覆盖/峰值内存/热功耗及视频证据。正式论文与当前公开适用载体已完整文献审查；这些未知并未变成独立复现成功。

待所有资源结项后统一筛选以下可证伪候选，交下一实验分支Gate3，不在本阶段执行：

1. 同权重逐转换载体检查布局、范围、边界与误差，分别控制全图、pad与tile；若误差主要来自协议，先排除实现混杂。
2. 隔离校准集，在同结构/数据/预算下比较浮点、PTQ与QAT，并记录实际手机委派和端到端成本；保留QAT仍损质量的反例。
3. 同一冻结输入视频比较量化前后空间质量、运动/曝光/噪声条件和时间/感知联合指标；控制InstanceNorm/分块/clip及量化阈值，不把代理下降等同观感改善。

本结论的主张级证据见[30项claims](claims.json)。Gate2另轮审查见[审查记录](../docs/gate2-review.md)与[机器可读决定](gate2.json)；通过后才可进入MIE-0025。0外部源码执行、0模型加载、0推理/训练/GPU/Gate3。
