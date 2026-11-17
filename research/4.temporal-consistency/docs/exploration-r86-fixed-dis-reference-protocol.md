# R86：一个固定快速经典DIS参考的完整质量—成本检验

2026-11-17；Codex  (sqs_A100)。登记先于任何DIS图像calc或新评价。R85显示当前half-FB成本，不能概括全部经典运动；note69核了原文摘要、OpenCV4.11 API与现库getter。本轮补传统参考，不是新方法、运动网络训练或为R74/R84等旧失败配置换flow挽救。原全部STOP保持。

## 唯一固定方法

原R44 square_mse两seed4075参数及原完整输入/输出缓存不变。DIS使用现有OpenCV4.11 ULTRAFAST全部预设默认（finest2、gradient12、patch8、stride4、mean-normalization/spatial-propagation=True、variational0），**原生尺寸**输入，不另做半分辨率resize、pad或改finest。这是一个公开预设pipeline，与原half-Farneback比较，不声称只隔离算法因素或复现原作者全部实现。RGB[0,255]round/clip uint8再灰度，back=calc(current,previous,None)、forward=calc(previous,current,None)，每次None，不传上一flow作时间初始化。复用两个DIS对象的内部工作区，每个序列重建；空间传播按默认保留。

native正反一致阈值严格1+.05*(back与采样forward范数和)，与原half-FB同公式和bounds；上一自身fused用原warp/bounds/mask fallback，alpha固定.5，首帧严格raw。部署只用当前/上一raw与上一fused，没有GT或未来。持久RGB状态与原rec同6HW×4B，DIS内部workspace另计为未测，不称完整RAM或零额外成本。

## CPU资格与完整质量

先一次人工资格：固定seed86001灰度随机纹理（128×160，经sigma.7 Gaussian平滑）组成RGB，current按已知native back=(3,-2)重采样，中心去16px后flow中位EPE≤1像素；另常量图一对只要求有限/bounds/首帧一致。共2双向对/4次DIS calc、0CNN。默认getter必须与上文一致。失败保存停止，不换纹理、容差、位移、精度或preset。

资格正常后仅原R30六SYN64帧+R10四native_crop ND16帧、两seed，复用同权重raw缓存，DIS共876双向相邻对/1752 calc，0增强CNN。原GT参考flow/mask/8px边界与evaluate函数不改，20新序列质量/896逐帧，保存自身输出数组至自有cache；每对部署接纳率单独记录。不重新估GT、不改ROI或忽略困难源。原raw/half-FB strong-rec40质量/JOD从R70完整表复用，原20 raw与20rec输出从R73引用清单读取，不用GT估计部署运动。

20新官方ColorVideoVDP0.5.7/standard_fhd/名义30fps/replicate，896比较帧，原适配器不改，0增强CNN。共享GPU只能在整进程前UUID绑定、真实CPU metric/首条件准备后再连续准入/锁、实际PID/UUID核验，优先空0–3卡、4GiB任务估计/2GiB allocator/既有任务增长保留、25%让步和逐项NVML监测；忙卡仅资源等待，不因观察超时重复科学任务。没有手机/MOS解释。全部20固定联系图（每seed每条件，t0/中/末，Input/GT/raw/half-FB rec/DIS rec）实际审阅另记，不以静态图判断视频闪烁或因果。

## 固定完整成本，不因结果挑选

为补经典效率前沿，质量/感知无论优劣都保留且测**同一轮**完整DIS-rec与原half-FB-rec（不测旧失败学习器、不扫preset/尺度）。两seed十条件前16帧，每模式2整段warmup+3正式、按轮次+seed交替顺序、每段空历史开始，合3200真实B1骨干（1280warmup/1920正式）；两rec各1500双向运动对，分别3000 DIS calc/3000 Farneback。DIS总预算为人工4+质量1752+成本3000=4756次calc；另原Farneback3000。没有新学习更新。

与R85相同resident CPU定义：HWC float32至tensor/255、真实CNN、clip/输出转换、各自真实运动/warp/融合、预分配sink复制均在wall内，解码/模型加载/sink分配/序列外核查在外。Torch intra2/inter1、OpenCV1、FP32/B1；CPU非独占，记录实际host/版本/亲和CPU数/load/进程CPU时间/每帧分段与首帧。只比较本轮匹配条件的三重复完整ms/frame中位数，保留全部120正式/80warmup与极值，不与R85跨轮数字拼成因果速度。R85 raw可作明确异轮描述性参考。每seed每条件首warmup分别与原half-FB rec缓存/本轮DIS质量输出前16核≤1e-5归一化最大差，共40资格、不增加CNN。失败停且不更改容差/线程/精度重跑。

## 完整判定与收口

四域-seed分别要求DIS对原half-FB：域PSNR差≥-.05dB、JOD差≥-.02、Etemp比≤1.05；逐条件最坏PSNR差≥-.30dB且JOD差≥-.10。不是要求已知传统方法原创或胜时序10%，而是是否足以作为更快近质量参考。完整成本要求每域-seed平均匹配条件DIS/half-FB中位时延比≤.80，同时保留20源分布。只有全部质量/成本条件都过才称该固定参考在本重复DEV的近质量更快基线；否则报告具体trade-off，不调阈值/融合系数/preset/分辨率/追加seed挽救，不代替创新方法、自然夜间、实际手机或独立最终。

CPU质量、完整成本及metric依赖各自在执行前冻结；不得改活跃源/科学目标，原normal或failure及原stdout完整保留，不以墙钟截止、科学自动重试或部分排名选择。结果/过程/解释/报告/各图分别文件级提交，阶段推送；自有data/project-python/main/指定作者/no-ff、所有旧STOP/保留最终/paper020与长期论文目标active保持。
