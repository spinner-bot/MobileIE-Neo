# O1正式学习及完整保留训练窗筛查

登记：2026-11-21；Codex  (sqs_A100)。原因：原CPU资格完整通过，允许检验共享主干/状态响应学习是否有独立视频收益。本版先于CUDA资格/正式更新/保留抽样评价；原25资格源和结果不改。基础相关、warp/KD/Sobolev已知，不先宣布创新。

## 数据、目标、学习预算

固定[原core/objectives](core.py)与[CPU协议](preflight-protocol.md)的5049参数两目标，全部原F0空间权重与matching种子9020+seed原样fresh构造；coordinate与response在每seed同初始化、独立Adam。CPU资格8更新和下一CUDA资格4更新全部丢弃，绝不继承模型/Adam或同另一seed结果。

seed0/1，MixedPool B4T4/128、2SYN/2ND，formal common_step0..3999对应data_step8000..11999，原两目标各4000×两seed，共16000正式优化/256000学生图；固定F0目标128000教师图，96000双向half-FB对。只用原低光与F0当前raw/其可观察clip估计场，不以GT生成部署输入或本次训练target。主干与新参数都可学，保留同原F0的起点不等于后续函数恒定。

两目标/空间项/质量mask/采样/公式原样，探针每共同步从prepare_teacher的seed参数draw_seed=seed+2*common_step生成（函数仅用此参数决定probe RNG，不改变教师模型/资料seed）。每步两张kernel5/17低通Gaussian状态，fresh draw；两组使用同一教师/批/探针，不在结果后追加频率/损失。Lspace=100*MSE未clip学生raw对F0raw；coordinate为native mask内L1/16+BCE，response为两个状态响应的教师能量归一MSE。不同损失尺度/FLOPs不当作纯信息因果。

独立fresh Adam(.9,.999)/eps1e−8/wd0/global clip1，原主干lr1e−5、新头lr1e−3，同时乘cosine倍率从1到.01：.01+.495*(1+cos(pi*s/3999))。FP32、TF32 off、确定性、CPU2/OpenCV1。保留1000/2000/3000完整资源恢复备份，只用唯一4000最终评价，不选best/seed/checkpoint/延长。资源暂停在两模型完整共同步边界保存模型/Adam/RNG/计数后让出，新准入后接同预算；非资源失败保留、不自动科学重试。无墙钟截止。

## CUDA资格和共享资源

单个seed0两共同步=两目标各2/共4丢弃优化、64学生/32教师图/24双向teacher对；与formal相同shape/objective，最终自身NVML≤4096MiB、allocator lifetime peak≤2GiB、实际PID UUID匹配、损失/梯度/参数有限。仅资格全过才fresh正式；资格loss不选设计。

同物理4–7组，最多四卡，本轮计划只用一张，允许有余量时并行共享。父11×2秒选择、进程前UUID绑定、子CPU准备后新11×2秒准入/项目锁，再实际PID设备校验。预计自身整进程4GiB、allocator2GiB，另给既有任务8GiB增长和1GiB安全余量，mean util≤60%、≥90% busy比例≤.25；逐批监测并让步25%占空。CPU准备后再次检查才用CUDA，不沿用旧准入；无余量只等/资源恢复，不杀/暂停他人任务或改功率。

## 完整固定保留抽样，不开启DEV

两seed最终后CPU各sampling_seed9140+seed、data_step16000..16003，B4T4，合32唯一训练场景窗口/128输入帧，非独立场景或DEV。每seed64F0图和两学生各64图，全384实际骨干图；每学生同时得到raw、描述/field，只有causal使用其历史状态。该共享调度不是独立模式推理耗时。逐帧PSNR统一事前120dB上限/逐项标记，原论文旧指标不改；SSIM/edge/共同GT估计half-FB残差时间MAE完整保留，无空域跳样。

每窗口13输出：F0 raw/自身half-FB .5/自身DIS .5三项；coordinate与response各direct当前/自身half-FB/自身DIS/causal/zero_flow_same_acceptance五项。zero使用自己的独立RGB历史，复用该学生真field的alpha和原bounds，只将采样改同坐标；是固定接受域下的位移干预，不是部署模式或免预测成本。两个经典post各自估计自己raw的field/独立RGB历史，不能拿GT或原F0的field替学生省成本。

合416窗口输出/1664逐帧/52域seed均值，8窗口/域seed/输出；完整GT/低光/输出缓存与逐窗来源保留。六条经典流（F0与两学生各FB/DIS）共576双向部署对，另96GT评价FB对；学生两种RGB状态轨迹共享field/编码，字段/warp调用另列。每seed首次窗口/各域末帧固定联系图和全表点图实际审阅，静态图不当闪烁/MOS/JOD、手机或独立最终。

## 预定继续门及止损

所有质量比较都须平均PSNR≥−.05dB、最差匹配窗口≥−.50dB，两域×两seed分别判断，不平均掩盖失守。

1. 两学生direct对F0 raw全部8项空间保护过。
2. response causal对自己的direct、zero_same_acceptance各四组须同质量保护、平均Etemp≤.90/≤.95参考，排除空间或无位移解释。
3. response causal对coordinate causal四组须同质量保护、Etemp≤.95，检验响应目标增量。
4. response causal对F0及response自身的half-FB/DIS共16组须同质量保护、Etemp≤1.05；只接近强post是必要筛查，不是论文已胜出。

全部36个比较同时通过才允许另登记完整DEV/JOD、独立外部与配对真实成本；手机仍待可用硬件/完整pipeline。否则停止固定共享8通道描述/两级soft相关/温度与confidence/响应probe/目标权重/LR/长度/seed局部网格，不以换teacher、加频率/GT loss或延长救分，不扩大失败候选成本/DEV/手机。所有局部正例和各门分项保留；不作全体operator学习/共享骨干不可能性结论。

代码、具名冻结、资格/完整结果、图表与审阅、决定/过程/报告分别小步提交、阶段SSH推送；自有data/main/作者/no-ff、所有旧STOP、原84值/snapshot003/PDF020/封存最终/7号和其他worktree保持。论文长期目标active，未以资格/负筛查代替方法目标。
