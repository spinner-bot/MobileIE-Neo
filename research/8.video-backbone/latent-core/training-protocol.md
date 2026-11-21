# L1：固定学习与完整保留训练窗投资筛查

登记：2026-11-21T10:54:26.123102+08:00；作者：Codex  (sqs_A100)。CPU资格已完整通过，见[决定](preflight-decision.md)；本版先于CUDA资格/正式更新/评价抽样。26资格源保持，原O1及所有旧STOP不变。L1是已知潜在状态思想的任务基线，不先主张创新。

## 固定起点、预算和目标

原core.py全不改。seed0/1各spatial/causal同初始化12431参数，分别fresh构造自对应R44 F0全部4075权重＋new seed9100+seed；独立fresh Adam，不继承CPU8或下一CUDA4资格权重/moments，也不跨seed继承。空间previous=e，循环previous=上一自由latent state，首帧=e。窗口内完整BPTT不detach，推理只有过去/当前，future0，不用GT/flow/teacher作为部署输入。

每目标每seed8000更新，四模型共32000正式更新/512000学生图，formal无教师图或FB。原各模型继承官方LoL预训＋项目F0 8K，另8K新训练，不称随机8K或同总预训练预算。MixedPool(seed,8000+common_step)，step8000..15999，B4T4/128半SYN半ND，两个目标逐步完全同资料；不读中途质量选择模型或改预算。

目标仅四帧所有像素未clip输出对GT的mean平方误差，无teacher/时间loss/额外噪声图/课程。主干lr1e−5、新参数1e−3，fresh Adam(.9,.999)/eps1e−8/wd0/globalclip1，cosine倍率.01+.495*(1+cos(pi*s/7999))。FP32、确定性、TF32 off、CPU2/OpenCV1，CuBLAS工作区:4096:8须在Torch/CUDA初始化前设置。只保留唯一新增8000最终；1000步完整恢复备份不选best/长度/seed，不设墙钟截止或loss早停。

## CUDA资格、资源与恢复

先同seed0/step8000两共同步，两目标各2共4丢弃更新、64学生图，另CPU原F0初始对照16图；原CPU8分列。初始GPU输出对同原F0 CPU≤1e−4，所有损失/梯度/参数有限，第二步shared head/project/gates/candidate/readout梯度非零。实际PID绑定UUID相符、自身末次NVML≤4GiB、allocator lifetime peak≤2GiB才通过；不是整进程lifetime峰值测量。不得继承资格状态。

同一0–3组，优先用户已授权的1/2号卡，最多四张，本轮最多两个seed各一进程；若忙可只用一张顺序执行。预估整进程4GiB、allocator2GiB，加既有任务8GiB增长＋1GiB余量，连续11×2秒检查free≥13312MiB、mean util≤60%、≥90% busy比例≤.25；进程前UUID绑定，CPU准备后再新11×2秒准入/项目锁、实际PID UUID复核。逐批监控与25%协作占空，让出只操作本任务，不改他人进程/功率；无余量只等。不因用户曾说空闲就跳过当前检查。

资源安全退出只在两目标完整共同步边界存模型/Adam/RNG/计数，释放卡，重新连续准入后接原预算；非资源异常保留停止，不自动科学重试或跳非有限step。源码/执行/评价准备齐并具名冻结后才启动。正式结束释放GPU后CPU评价，不占卡等绘图。

## 共同固定抽样与完整输出

两训练seed使用**相同**评价draw：sampling_seed9150、data_step20000..20007，各B4T4/128；32唯一窗/128唯一输入帧在两seed重复评价，16窗/域/seed。相同抽样便于区分模型种子和窗样本，不构成独立场景。只有原57 DAVIS和12实拍训练源，不开DEV/保留最终。所有输入/GT/输出和逐窗来源保存自有cache。

每seed/窗11输出：F0/F0_FB/F0_DIS三项；spatial_current/spatial_FB/spatial_DIS三项；causal/causal_current/causal_self4/causal_current_FB/causal_current_DIS五项。所有评分输出clip[0,1]，latent递推仍用原未读出state；FB/DIS各自只从对应clipped raw估计自己的几何/上一fused RGB，固定.5，不给GT辅助。causal_current逐帧previous=e，与独立训练spatial区分。self4每帧独立从state=None开始，**同一low连续调用完整模型四次并传自己的state，最后一次raw输出**，不读真实历史；不缓存重用head/省掉中间原backbone，因此四倍网络调用明确计入，它是重复计算的强诊断，不是等成本因果隔离或部署主输出。

每seedF0/spatial/causal/causal_current各128骨干图，self4另512，共1024；双seed2048骨干图，其中F0256、学生1792。共704窗口输出/2816逐帧/44域seed均值；经典三条raw各两种flow×32窗×3相邻×两seed=1152双向部署对，另192共同GT评价FB对。没有GT或teacher reset模型状态。共同GT估计half-FB残差时间MAE/8px边界/PSNR事前120dB上限及逐帧标记、SSIM/edge沿原定义，全样本/首帧保留，不跳空域。

每seed第一draw step20000、window0/2的末帧固定联系图，包含GT/low/全部11输出；全44均值、所有配对和门图实际审阅，单独PDF导出。静态图不当视频感知/MOS/手机证据，本筛查不启动JOD。

## 不同收益路径的预定门

所有比较各域×seed分别满足平均PSNR差≥−.05dB、最差配对窗口≥−.50dB，不只报跨seed平均。44个唯一比较为8空间、12历史和24强post；强post另有A/B两个固定时间限，不在结果后放宽。

1. 空间保护8项：spatial_current与causal_current各对F0，四组均通过质量下界。
2. 独立历史12项：causal分别对独立spatial_current、同权重causal_current平均Etemp≤.95；对同权重causal_self4≤.97，均同时过质量保护。self4使用更多计算，不能由它差推断唯一失败原因。
3. 强post24项：causal分别对F0_FB/F0_DIS、spatial_FB/spatial_DIS、causal_current_FB/causal_current_DIS，四组均过质量保护。A要求Etemp≤.90，B要求≤1.02；A/B还各需前两条全部通过。

A/B均失败则STOP固定32ch/d4-d8/门/LayerNorm/读回/初始化/GT MSE/目标权重/LR/长度/追加seed网格，不换teacher、加flow/监督/更长窗口或频率救分，不扩该失败候选DEV/JOD/外部/成本/手机。局部正例与SSIM/edge完整保留，不称所有latent RNN无效。

只有A或B质量资格通过才另登记完整CPU配对成本：A整链≤所有对应强post1.05倍；B整链≤所有对应强post.80倍。B是质量近似/更低成本候选，不能称比强post少10%闪烁。成本通过之后仍须另登记独立DEV/JOD及外部/手机验证，不自动解封最终。即使全过也只是已知思想基线信号，创新仍要相对RLSP/GRU-VD/Turtle等核查具体差异，不能把可运行组合当论文方法。

所有源/资格/完整结果/图审/决定/过程/报告分别密集提交，阶段SSH推送research/O1，未来合并--no-ff不rebase/squash/force。资产自有data、作者Codex  (sqs_A100)、原84值/snapshot003/PDF020/独立最终/7号/并行main与其他worktree保持。论文目标active且远未完成。
