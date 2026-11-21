# MIE-0024 QATIE：逐节阅读与实现核查

2026-11-21；Codex  (sqs_A100) <langxibielangle@qq.com>。有效阅读序27。本文为当前资源的过程记录，尚须另轮Gate2；不修改冻结标准或已结项资源。

## 阅读与材料身份

- arXiv 2604.21743v1（2026-04-23）十个物理页全部文本、36条参考文献已读。正式CVPR2026 Workshops/MAI，June2026，3847–3856，十页PDF完整23724214B；完整172行版本diff已读，科学文本、公式和表值一致，区别为首页通信邮箱/格式、出版水印与页码。不是两次独立实验确认。
- 预印本第4、6页以及正式第4、6、7、8页实际查看，覆盖全部3图5表；不是每个纯文本页都原尺寸查看。作者两PNG均实际查看；透明图第一次直接RGB转黑底隐藏箭头，另生成白底预览重看，原始RGBA及两预览均保留，不把渲染问题归于原图。
- 官方GenAI4E/QATIE固定6703a6d123042601a465378a74bdfe224696fe7d、tree ac4bf360ee5dcf636915f3a11486d14c003dc878、提交2026-06-04。全部37文件29145635B已逐Gitblob/ZIP/SHA核；26适用文本全部静态阅读。requirements原UTF16LE，按BOM解码侧存再读，不把UTF8失败当文件损坏。
- 26文本包括两README、MIT许可、依赖、四shell、模型、loss、训练与scheduler、两data文件、checkpoint/loading两入口、三eval与两benchmark、两个batch导出、to_tflite、quantize及infer。逐文件SHA见reading-progress.json。不是仅根据脚本生成的平文本声明“已读”。
- CKPT仅ZIP CRC/pickletools；992成员、986storage、10987688storage字节，不反序列化。1186个model键文字含1040 observer键；可见epoch37/global_step47652/Lightning2.6.1文字及down1权重32×3×3×3附近操作码，不证明完整训练历史或训练配置。无hyper_parameters文字不等证明训练未保存任何其他元数据。4个PYC只做完整留存/清单，不import或marshal。
- 两TFLite仅按官方TF2.20.0 FlatBuffer schema由自有stdlib脚本读元数据，不初始化解释器。schema原件、检查脚本、所有原始文件和摘要都可解析。0作者代码执行、0模型加载/推理、0训练/GPU/Gate3。

## 论文逐节

**p1 摘要/引言。** RGB手机到DSLR的配对照片增强，统一噪声、颜色、模糊和tone；不是RAW输入、视频或专门低光任务。把训练与整数部署偏差作为主问题。摘要“防止质量下降”须以表5剩余损失约束，不能照收为无损量化。

**p2 相关工作。** 学习ISP、有效结构与量化为先例；并无历史帧/流/显式时序模块。不能因手机部署目标自动归类实时视频方法。

**p3–4 方法。** DaHua-IIG架构明确引[16]（已入目录MIE-0033），门控编码的两个stride2卷积/tanh输出相乘；两分支原特征cat作skip。在S/2、S/4、S/8精炼，decoder融合上采样、精炼encoder、rawskip三流，1×1压缩后精炼。图1显示InstanceNorm和残差分支。普通门控/concat/UNet不是MobileIE MBRConv继承，也不是本二期可直接申报的新机制。

**p4–5 损失。** MobileIE[35]具体在outlier-aware及总损失、p7 loss消融；p10准确书目ICCV2025 21949–21960。Lpsnr=(50−PSNR)/100，原文界限并非严格截断：PSNR>50时为负。当前实现训练PSNR转Y，余两项为RGB全局flatten余弦和spatial离差相对加权L1。权重有界、detach，偏离当前误差均值越远权重越大；不简单称“压低离群点”。当误差为统一常偏差时该项权重为零，但其他loss仍作用。本文借鉴的是损失先例，不是MobileIE骨干。

**p5–6 训练/评测。** 作者报告DPED iPhone3GS/Canon训练>160k个100×100patch，50ep、等效batch128/accum2、A6000及约62k步；评测9310多手机patch。FullHD成本与照片patch质量不同尺寸。比赛表1明确1024×1024，Capybara21.82/.7653、MOS3.2、GPU291ms/Arm266ms、score3.8；不可把21.82直接同22dB的另一验证表门槛混用。不同队伍MOS、chip路径与训练预算也不构成我们独立公平排名。

**p6–7 图3/结构消融。** 一张街景及车牌裁图中PTQ天空偏粉，QAT较中性但仍可见噪声/细节差别；选图不代表所有照片或人类评分，更不是视频播放。全模型22.194/.796/469ms；无res22.029/.793/464、无fusion21.940/.793/396、无resolution20.398/.789/237，精炼改善质量也增加成本，不能只报质量列。结构替换改变容量，不能证明纯机制效应。

**p7 宽度。** c16/24/32/64参数.230/.516/.915/3.651M，时延180/249/469/1432ms；c24作者score最高，c32质量取舍不是一切目标下最优。加宽不自动扩大几何感受野。当前源字面计数可对应这些四舍五入：230083/515811/915331/3651331；不是和MobileIE四千参数同预算。

**p7–8 loss。** 本loss 1.67h、22.194/.796，对照8.58h、19.79/.8763，后者SSIM更高，不是全部指标占优。当前MSSSIM类只实现一层SSIM，未做多尺度下采样；不能从类名认定重现了论文MS-SSIM对照，历史实验版本未知。

**p8 表5量化。** 正文称FP16，表头FP32，两处原样保留。FP32 PTQ22.358/.794，QAT22.194/.796，GPU均469ms/HTP151ms；INT8 PTQ20.576/.6139、GPU319/HTP41.4；INT8 QAT21.050/.725、GPU319/HTP41.8。QAT对INT8 PTQ为+.474dB/+.1111SSIM，但较其FP32 QAT仍−1.144dB/−.071。HTP151/41.8≈3.61倍，只是作者条件下比值；41.8ms倒数约23.92FPS，不是30FPS或端到端持续视频证据。QAT也未比PTQ的41.4ms更快。p9–10结论/36参考已读；不因参考关联无限扩展视频增强全领域。

## 当前源实际路径

**model_builder 1–450。** 35Conv、5InstanceNorm，nearest上采样、最后clamp(x+delta,0,1)。InstanceNorm默认路径按当前图像空间统计，分块可能改变结果；并无输出帧间状态。QAT qnnpack、head明确FP32，ConcatFP32默认qconfig=None；legacy选项改变准备图。不同阶段FP32/INT8 carrier须区分，不能从这个设置宣称最终TF文件全浮点。

**loss 1–297 / train_utils 1–78。** loss1 PSNR+命名MSSSIM+outlier，loss2（论文推荐）PSNR+cos+outlier，loss3才有ImageNet VGG19；不得把所有loss都说成外部VGG教师。PSNR_Y与RGB评测不同；MSSSIM实际单层已核。outlier用delta=out−lab、空间std/√2和均值、tanh相对距离.detach。Cos为全RGB展开，非局部/跨帧对齐。

**train_qat 1–351 / data_aug29 / dped_dataset137。** CLI默认c24/loss1、50ep/b64/accum2、limit_train_batches=.1；README示例c32/loss2/limit=.01，不等论文完整预算记录。QAT会把precision强制32，不能仅据BF16示例认定实际QAT算子BF16。可选pretrained_ckpt，没有默认自动完整两阶段；默认LR1e-4而论文QAT描述1e-5，须固定原命令。默认norm clip1.0，不等逐元素[-1,1]。训练step输出加1e-6Gaussian不用于推理。只iPhone训练，test作val；路径glob独立且未排序/核文件名，实际配对须另核，尚未观察到错配。浮点np→ToTensor再/255正确，不能误称重复归一化。patch_data='False'字符串仍真是条件路径问题；默认True不是因此失效。可选input-only颜色/gamma/gain、paired翻转，没有时序增强。未执行W&B/下载/训练。

**checkpoint/loading/export-utils 全读。** 多处strict=False并报告missing/unexpected，备选ablation按键匹配后使用名称/元数据；无残差开关与base权重键相同，不能仅凭CKPT识别历史语义。旧observer键可移除给裸FP32 ONNX；不表示训练过QAT的参数失效，但部署scale/graph身份需核。我们的CKPT完全不加载。

**to_tflite295 / quantize337 / batch导出386+228。** 一条是FX QAT convert_fx，可复用checkpoint observer；FP32 warmup读iPhone测试集，不用GT却仍接触测试输入。另一条拆出浮点权重→ONNX→onnx2tf→TF PTQ，默认128测试输入代表集，再用该测试集评价，未隔离校准/最终评价；不把接触输入说成用了测试标签优化。dynamic wrapper用reshape(predictions,shape(input))而非transpose：若原TF输出NCHW，不能自动等同NHWC，仅列条件风险，不声称公开大文件由此生成或已坏。ONNX默认18、固定100×100；FP32允许SELECT_TF_OPS，INT8限制BUILTINS_INT8及signed int8I/O。

**TFLite两原件静态元数据。** 100版1005112B：输入FLOAT32[1,100,100,3]、输出FLOAT32[1,3,100,100]，229tensor/139op，185INT8/42INT32/2FLOAT32，入口QUANTIZE/末尾DEQUANTIZE。大版1002624B：输入输出INT8[1,1920,1080,3]，216tensor/128op，176INT8+40INT32，无float typedop，scale≈1/255、zero−128。大版H=1920/W=1080，不把文件名默改成landscape。二者不只是空间尺寸变化。signedINT8与论文UINT8活动量的差异可能通过仿射映射表达同范围，不能据符号直接判质量错。静态类型并不证明QNN全委派、运行成功、权重表身份或端到端性能。

**infer_tflite353。** 做量化/反量化、输出CHW→HWC检测、对齐reflect pad/crop，先尝试原图resize_tensor_input，失败回固定tiles；另有显式resize策略可能改变比例。tile线性mask虽注释raised-cosine，linspace端点为零，在overlap>0的强制分块/回退外边界累积权重为零，最终可能给黑边；这是源码条件推断，未跑模型，native图匹配路径不能据此全盘否定。InstanceNorm又使tile结果未必等于全图。

**eval_tflite488 / eval_pytorch115 / eval_original276 / benchmark340+570 / 四shell。** patch loader和fullHD loader不同：前者sorted配对/计数但仍无逐basename核，fullHD按basename找多手机；原DPEDData仍独立glob。TFLite评测先resize输入/GT至固定图形，CPU interpreter默认无delegate；计时含输入量化/输出反量化/invoke，未含图片decode/resize、warmup、手机同步等。PSNR阈值1e-7→100，PyTorch1e-10→100，train valisclose另一路；TFLite SSIM显式population covariance，而PyTorchskimage入口未显式固定同项，均8bit截断后RGB逐通道，不能假设完全同定义。只把原语义记下，不宣称影响已报22dB。

batch FP32导出每CKPT一次CPU TFLite评价，按请求生成cpu/cuda CSV行；INT8批导出每行再调用CPU TFLite，标签cuda仍不变成GPU运行，源文档亦说明此事。benchmark_ckpts才是Torch设备实测路径，默认随机1088×1920、20warmup/100runs/5trials，非论文手机QNN结果。INT8 benchmark的Torch分支没传full_hd，而TFLite分支传了；原图评测注释宣称各手机native但当前BlackBerry/Sony配置2560×1440而非注释3120²/2592×1944。这些是未来运行前要显式锁定的协议，不是我们执行过的实验错误。

## 成本与可迁移性

静态层求和P(c)=889c²+156c+3，c32=915331；仅35Conv的1080×1920 MAC约144.156672G，不含norm、激活、插值、concat/搬运和量化；完整内存/时延未测。c32 S/2的4c逻辑concat有66355200元素，FP32约265421KB、INT8约66355KB（十进制），不是实测峰值或TF分配器结果。初手算线性256被逐层assert拒绝，复核得到156、各宽度匹配论文，错误未写成论文差异。

可迁移的是：固定输入范围/布局/observer/校准集/图形版本，保留FP32–PTQ–QAT三者质量与端侧完整成本，以及所有不利结果。不能从QAT直接推视频稳定；动态图像统计、clip/量化阈值、曝光变化可能产生跨帧敏感性，但尚未证明本模型闪烁原因。三组建议均交Gate3：同权重逐carrier数值与边界、隔离校准的同预算量化控制、真实运动/曝光/噪声下质量及时间/感知联合控制。本阶段不运行。

缺口：原始train命令/seed与数据ID/分组、原数据许可与对齐、每表对应权重/转换版本/手机文件SHA、完整每图/每trial结果、delegate覆盖/峰值内存/热功耗/端到端视频。没有单独承诺却未取得的公开补充；充分公开文献分析仍不等原论文复现。新资料只追加复审。
