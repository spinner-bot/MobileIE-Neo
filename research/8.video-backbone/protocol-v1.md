# V1：固定低计算空间骨干与完整因果观测融合

2026-11-20；Codex  (sqs_A100)。正式训练前登记；组件/对照见[设计与先例](precedents-and-candidate.md)，原训练来源见[继承审计](original-model-and-training.md)。这是系统可行性研究，不是已确立的新方法。research/7不使用。

## 唯一候选与对照

实现 `packed_backbone.py` 和 `causal_pipeline.py`，固定4×重排/64通道/两DW块/残差平方/HDPA/残差RGB；空间16180参数，学习门后16365。三组为独立训练spatial、原生低光状态fixed(.5)、learned(185参数门)。共同从同seed唯一教师暖启动复制骨干，fixed/learned在首更新前alpha相同；末层门零初始化，不能要求其上游参数首步就有非零梯度。所有空间权重参与训练，不冻结F0后补小头。

因果模型每帧先独立preview，再用当前/前preview灰度估计原固定DIS ULTRAFAST与FB接受域，原生连续warp上一低光融合状态后重建；首帧直接preview。没有未来帧、GT运动教师或GT部署输入；运动计算stop-gradient是设计的一部分，train和infer均使用自身preview，不能缓存固定F0运动替代。窗口内状态保留BPTT，窗口间重置；验证/推理detach跨帧图但不重置数值历史。

保存状态是3HW FP32低光＋HW uint8灰度，名义13HW字节，1080p约26.96MB（十进制），不含OpenCV工作区、GPU暂存、sink/输入或参数。经典post也能只存灰度，必须按同等优化报告。两遍CNN不叫零推理开销；对齐/转换/门/重排/复制完整计时。

## 数据、初始化与固定算法预算

只复用既有MixedPool：B4、T4、128裁块、2 DAVIS合成＋2 OLiVES真实配对训练窗，保持原空间翻转/退化/排除源；不打开最终OLiVES43/71/83、DAVIS testdev/Kodak保留。训练seed0/1，各自step0..5999，不从DEV选择数据/终点。

每seed先全新packed core，seed=8900+seed，**2000更新纯输出KD**，data_step0..1999；教师为同seedR44 square_mse唯一最终，冻结eval、输出clip[0,1]，训练MSE只对其输出。教师继承官方LoL预训练及项目内8K更新，不等于新模型继承了同样的优化历史；计入64K教师图像前向。warm LR1e-3；全部FP32/TF32关闭，Adam(.9,.999)/eps1e-8/wd0/globalclip1。

共同暖权重复制到三组，每组**4000 GT适配更新**，data_step2000..5999，fresh Adam1e-4，其余同上。spatial loss为四帧MSE；fixed/learned为`.5*MSE(preview,GT)+.5*MSE(causal,GT)`，首帧复用结果不额外调用。raw输出参与loss，不先clip；没有额外时序/variance/teacher/flow loss。循环两组目标和优化设定相同，空间组的独立目标差异披露，不能称相同训练FLOPs。

总正式更新28000 = 两seed×(2000＋3×4000)。warm packed/教师各64000图；适配独立空间128000图，两个循环组448000骨干图，共新骨干640000图（含warm）。正式DIS192000双向对/384000calc；门96000图。CPU/CUDA资格和最终重载单列，不混正式训练计数。每1000步备份，只有最终2000warm和4000adapt进入评价，不择中途/seed。没有墙钟截止/追加预算/科学失败自动重试。

## 有限实现资格与共享资源

CPU2/OpenCV1，先人工奇数尺寸检查尺寸/有限性/初始identity/解析参数和卷积MAC；再仅seed0 step0首实际训练窗，warm两步丢弃，三组各两步丢弃，合8；检查全部活动梯度存在/有限、warm后固定和学习初始输出一致、首帧等于preview、循环末帧对过去low有梯度而spatial为0、保存/重载有限且输出一致。人工检查是工程资格，不作为方法实验或平移等变证明。全部丢弃，不继承。失败原样记录停止，不放宽阈值/换形状寻找通过。

CPU通过后用同一数据/预算做一次CUDA资格8丢弃；完整CPU池/teacher/model/首窗准备后才新11×2秒采样并持项目锁。整个进程在Torch导入前由既有 `scripts/prebind_shared_gpu.py` 绑定，核实际PID/UUID；同组0–3优先一至两卡，≤4。预估每进程4GiB/allocator上限2GiB，既有任务保留8GiB增长＋1GiB，平均util≤60%且≥90%忙样本≤25%；合作占空目标25%，逐更新监测。资源暂停仅已保存完整更新边界，可原算法步数续接；不改其他人任务或功率。资格完成后重新初始化正式任务，再新准入；不能复用资格优化器。

## 唯一最终开发评价

全六SYN条件×64帧＋四ND条件×16帧，即每seed448帧，既有DEV输入/GT/指标接受域；训练域来源、重复DEV及合成限制保留。CPU2/OpenCV1完整视频流，所有输出clip[0,1]计指标。每seed：spatial raw/自身half-FB .5/自身DIS .5（3）；fixed causal/独立preview/该preview两个post（4）；learned同样4。共**220质量行/9856逐帧/4440新骨干图**，11输出×10条件×两seed，全部唯一最终；另复用原F0 raw/两个经典60行，完整280行。每个preview的post使用其自己估计运动和自己的RGB融合历史，不能复用GT部署场。

预测数组自有cache；全部220输出用官方ColorVideoVDP0.5.7、standard_fhd、既有名义30fps评分，新的metric GPU按实际尺寸重新准入，不借训练峰值。完整所有域/seed/源公布PSNR、SSIM、Etemp、JOD、参数/MAC、模型/状态存储。固定四张联系图：seed0/1×bear_photons20、video_47，各首/中/末；必须实际阅图，静态图不代替视频闪烁或MOS。

质量保护先看独立spatial相对F0 raw：每域每seedPSNR≥−.10dB、JOD≥−.05，逐条件PSNR≥−.5dB。对learned causal的主要参考为：旧F0 half-FB/DIS、独立spatial half-FB/DIS、同权重learned preview half-FB/DIS、fixed causal，**共七组，全部报告**。每域每seed及每参考PSNR≥−.05dB/JOD≥−.03，逐条件PSNR≥−.5dB/JOD≥−.15。

A：上述质量保护全部成立，且对七参考每域每seed Etemp≤.90；B：相同保护、对七参考Etemp≤1.02，并对fixed causal Etemp≤.95，表示效率取舍而不是时序10%优势。若A或B质量合格，另登记相同16帧prefix、所有条件/seed、共同CPU执行的完整延迟/峰值工作区比较，保留warmup2次/正式3次及顺序交替；A还需成本≤最强对应经典1.05，B≤.80。解析MAC不能替代此门槛；实际手机另需硬件，未测不声称。

固定循环控制即使学习门失败，若相对空间与其自身post出现符合相同质量/成本门槛的价值，也只登记为已知组合的工程可行性正例，**不把它升级为学习门方法成功**。若独立空间保护或A/B均失败，停止本固定packed核/门的宽度、phase倍率、alpha、噪声图、loss、LR、长度、seed网格；不追加旧R20/R26/R44/R51/R64/R74/R88救分。此结果不证明所有低分辨率骨干/递归恢复不可能。

## 记录与边界

运行全部在zrh自有data与project-python；科学源码/协议正式前冻结，配置/原始结果/失败/解释/过程/报告分别小步提交，阶段SSH推送，main/no-ff/指定作者，其他worktree不碰。有限资格/完整学习/所有开发评价不等于可验证创新、自然动态泛化、实际手机或独立最终通过；长期论文目标保持active。
