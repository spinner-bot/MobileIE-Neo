# R82共同条件修正的固定正式检验（尚未启动）

2026-11-17；Codex  (sqs_A100)。CPU原型资格已通过12丢弃更新；本登记先于CUDA资格/任何正式更新或新DEV输出。通用核混合已排除为新贡献，只检验保留当前函数的参数化是否有可用时间收益。当前正式trainer/evaluator尚需实现与冻结，不把登记称启动。

## 模型与可证伪问题

原F0固定R44 square_mse同seed 4075参数/累计8000，eval/冻结，当前c=clamp(F0(x),0,1)。三个独立H按prototype原模块：Conv1(6→8)+PReLU8+DWConv5(8)+PReLU8+无bias Conv1(8→3)，304新增/4379总参数、最后权重0初始化。spatial=c+H(c,c)；direct=c+H(c,h)；centered=c+H(c,h)−H(c,c)。每段首帧三者均c；后续h为各自前一输出clamp，训练窗不detach、部署仅过去；spatial只需首帧标记。不存在GT/未来历史、flow或新增外部encoder。

centered相同输入始终等于F0，只排除不依赖历史的加性空间修正，不排除“不等于当前”的开关或其它捷径。不得由self恒等式/梯度非零宣布时间信息有效。H线性是常见创新项，冻结底座/残差/卷积有先例，当前无确立的新颖性。

## 资格、预算与优化

同MixedPool 57 DAVIS训练源/12配对实拍、原50/50 B4T4×128；每seed固定data_step8000..15999，三组共享同一输入/GT/冻结F0结果，各自H/状态/Adam独立。两seed三组各8000，共48000正式optimizer更新；F0只共享前向，不更新。合64000次B4骨干批前向/256000图块，H基本192000次B4调用/768000图块等效，检查点/诊断另计；不复制F0给三组重算。

fresh Adam lr1e-3/betas(.9,.999)/eps1e-8/wd0，恒定学习率、globalclip1，FP32/TF32关闭、确定性。同四帧clipped RGB MSE含首帧等权，不加辅助/时间loss/flow伪标签、不改曝光、BN或原FST。协议选定优化不作原KPN复现；CPU资格SGD不继承。1000步保存完整三H/各Adam/进度，唯一8000最终选模，不按DEV挑checkpoint，不增长度/seed/核/通道/LR/loss网格，无墙钟截止或科学自动重试。

CUDA先以同CPU资格的seed0/step8000窗口、两个F0/三控制各两次丢弃SGD，共额外12，核CPU同等函数/current/梯度/有限状态与流式≤1e-6；实际峰值/完整进程显存记录，全部丢弃后fresh初始化正式。新的全部正式源/依赖及完整CPU/感知评价链需先实现/编译/冻结，CUDA资格不等于正式结果。

共享GPU按用户允许有余量共卡，同组0–3最多四张，优先空卡；父进程前UUID绑定、CPU数据/模型/首窗准备后的11×2秒新采样与项目锁、实际PID/UUID核查；预计进程增量4GiB/allocator2GiB，为既有任务保留至少8GiB+1GiB增长余量，25%协作让步并持续监测，不改别人任务或设备设置。两正式seed最多各一卡。资源不足只等待/本项目安全边界让出，不因观察超时重启；原实现失败保留、无科学重试。

## 唯一完整共同评价

最终两seed正常后，既有R30六SYN64帧（bear/lucia/walking×ph20/80）+R10四ND native16帧，不打开保留最终。使用同seed F0既有clamped raw预测数组与来源校验作为公共冻结输入，CPU2/OpenCV1；这避免重复骨干，但实际部署必须计算F0，不能称0CNN成本。三新模型各一完整因果序列；spatial再原自身raw half-FB递归.5；centered再错误历史诊断，共每seed50新条件/2240比较帧、合100新质量/4480帧；复用原F0 raw/strongrec40条件，联合140条件/28域-seed均值。

centered_same逐帧实际计算、检查恒等并alias原raw，不新增重复质量行。仅模型H原生图像调用合7008：两个seed各438非首帧，spatial1/direct1/centered2/wrong2/self2。新spatialrec只自身输出估计flow，不用GT部署对应；所有质量指标仍使用既有GT评价流/接受域，不改变ROI。保留全部新输出/来源/逐帧，不挑源或最好seed。

错误历史预先固定：先保存centered真实因果输出；同域/photon条件按bear→lucia→walking→bear循环、ND按video_15→47→91→121→15循环，注入另一源t−1真实centered输出，首帧仍F0。是外源teacher-forced诊断，不递归错源、不用未来/GT，不是部署方法；同尺寸/长度先验检查，不临时缩放/选相似源。wrong差异可能含分布影响，不能单凭它证明全部信息机制，但无正确历史优势须否决时间贡献叙事。

新100输出完整ColorVideoVDP0.5.7/standard_fhd/名义30fps/replicate，4480比较帧，旧F0两参考40JOD复用；CPU正常后重新GPU准入、无增强网络/SGD。20固定首/中/末联系图，Input/GT/F0 raw/强rec/spatialrec/direct/centered/wrong(诊断)全列保留并逐张实际查看，不作MOS。即使CPU不利也完成这次预定完整指标/图链，不看部分排名改代码。

## 固定停止线与进一步投资

只以centered真实因果输出为候选，每域/seed分别比较六个参考：F0 raw(同current)、spatial raw、direct raw、centered wrong、原F0 strongrec、新spatial strongrec，共24比较。全部要求PSNR差≥−.05dB/JOD差≥−.02；前四类Etemp比≤.95；两强rec A要求≤.90，B资源资格允许≤1.05。所有源PSNR相对两强rec最坏差≥−.5dB，正负条件全报。

A还需实际完整CPU部署中位耗时≤两强rec各1.05，才有相对时序改善的成本资格；B需所有强rec时间≤1.05及前四历史/容量门槛，并实际完整CPU中位耗时≤两个强rec各.80。B是资源取舍，不能称时序赢10%。手机仍需后续实际测量，CPU不代替手机；完整pipeline计入F0/H/状态拷贝/flow参考等，不以参数或理论MAC认定效率胜出。若A/B均未过质量资格，不为其扩成本/外部/手机实验。

若任何必要历史/容量/强rec质量门槛失败，停止本H宽度/深度/核、相减幅度、输入、目标、学习率、长度或seed局部网格，不叠旧STOP/外部flow/post挽救。不把普通direct/spatial改善或归档数量当方法成功；通过也需独立先例差异/机制、自然动态/手机/独立最终，保留像素不自动解封。所有资产只zrh data、project-python/指定作者/main/no-ff/密集分别提交/阶段SSH推送保持。
