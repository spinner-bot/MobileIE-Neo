# 主干信息流与时域化

2026-11-20；Codex  (sqs_A100)。main主题8，research/7继续留给其他分支。

当前[M1新骨干](multiaffine-core/README.md)已实现、CPU/CUDA资格通过并启动空间恢复必要检验：[0188](../../docs/进展报告/0188-T-26112023-多线性骨干实现资格与空间恢复正式启动.md)。5340参数、解析1080p2.712269GMAC、FP32缓存99532800B；raw独立角点/来源与stream检查通过，但clip反例存在，非最终RGB均值保证。新随机M1/原Slim各8K×两seed=32000正式更新/512000学生图，GT MSE/40冻结、物理6新准入/UUID匹配，首正式worker3394474和父3393743的原start已核；CPU4/CUDA4更新全丢弃。首观察各472更新，完整恢复结果未得；只有八个恢复保护过才另登记视频。所有旧STOP/最终/原论文值保持，论文goal active。

此前[N1三观测方差资格](triple-variance/README.md)已完整结项：[0187](../../docs/进展报告/0187-T-26112023-三观测方差识别校准与骨干问题收敛.md)、[决定](triple-variance/decision.md)、[真实终态](triple-variance/process/terminal001.json)。双冻结F0/四训练纹理、11264NN正常111.089698秒、0更新/GPU/DEV/最终；24独立主组12个完整通过，固定photon8/8正例保留，跨photon4/16，完整实际运动资格False。停止固定三观测/16tile/截断/窗口/模式/相关构造网格，不直接接噪声图或训练补救。全48组和24主组两PNG实际查看，原进程已无；[后续骨干问题](triple-variance/next-core-question.md)仅待核跨层不重叠观测的多线性来源树，不是已实现/登记模型或已确立创新，论文goal active。

此前[S1单向传输与贡献者离散度](single-direction/README.md)已完整负结项：[0186](../../docs/进展报告/0186-T-26112022-单向传输贡献者离散度筛查与止损.md)、[决定](single-direction/decision.md)、[原真实终态](single-direction/process/terminal001.json)。训练窗双冻结F0，64NN/16窗/80条件、0更新/GPU/DEV/最终，正常38.999856秒；对原FB三个组PSNR低.51–.73dB、ND0有小正例，三个完整门均False。对原FB仍9/16窗口联合正、对raw12/16联合正。停止固定目标贡献者矩/mass界/阈值/alpha/preset/field来源网格，不训练补救或扩失败DEV/成本。全20点及16窗口图实际查看，无活跃作业，论文goal active。

此前[P1粗能量历史零更新筛查](phase-envelope/README.md)已负结项：[0185](../../docs/进展报告/0185-T-26112022-粗能量历史零更新筛查与止损.md)、[决定](phase-envelope/decision.md)、[真实终态](phase-envelope/process/terminal001.json)。训练纹理已知位移、四实际路由、双冻结F0，1280NN/0优化/200输出条件正常32.238270秒；对raw四噪声组时间升.456%–.560%，对同尺度feature EMA全部32 noisy条件时间更差，完整训练资格False。clean仍有两seed时间降18.8%–22.8%/质量正例；不抹掉局部收益。直接Riesz/幅度滤波已有先例，非确立创新；停止固定能量/差分/alpha/尺度/clip/epsilon网格，不训练补救/扩DEV。全30均值图与固定联系图已实际查看，原任务全退出、无活跃GPU，论文goal active。

[T1同原骨干训练监督对照](teacher-targets/README.md)也已完整负结项：[完整决定](teacher-targets/decision.md)、[0180报告](../../docs/进展报告/0180-T-26112019-同原骨干时序KD完整评价与止损.md)、[原终态](teacher-targets/process/terminal001.json)。24000正式更新/六唯一最终/全180质量与180官方JOD/240共表及四图实际审阅，恢复保护3/4、A/B各0/40、成本资格False。对空间KD全部20条件时间更好，四组改善1.49%–2.38%，但距原强post仍高约29%–48%；固定同F0/T4DIS/等权KD局部网格STOP，不扩失败成本/外部/手机。普通KD是已完成已知参照，不主张创新。全部原任务终态、GPU释放，论文goal active。

[T2部署递归监督](recurrent-objective/README.md)已完整负结项：[0184](../../docs/进展报告/0184-T-26112021-部署递归监督完整评价与止损.md)、[决定](recurrent-objective/decision.md)、[真实终态](recurrent-objective/process/terminal001.json)。全16000正式更新/四唯一最终/120质量与120官方JOD/360共表/72均值、四图与两全点PNG实际审阅完成；恢复0/4、A0/44、B10/44、额外5%学习0/8，完整资格False。对clip-GT DIS有SYN PSNR+.10–.11dB/14条件PSNR正例，但全部20条件时间更高（四组+1.50%–2.46%）；固定原F0/native DIS/.5/四帧递归监督局部网格STOP，不扩失败成本/外部/手机，不继续向T1/T2加loss或延长诊断。所有原任务终态、GPU释放，已知递归训练非创新。后续须选择实质不同信息/结构问题、先核先例再有限登记，不预先称另一模型有效。


V1已完整结项为负结果：[决定与止损](decision-v1.md)、[0175报告](../../docs/进展报告/0175-T-26112017-完整视频骨干候选评价与止损.md)、[原终态](process/v1/terminal001.json)。两seed全28000正式训练/八唯一最终、220质量/220官方JOD各9856逐帧、280共同条件/56均值及四固定图均正常，四联系图和两全表图已实际查看。全部旧worker/coordinator均退出，无活跃V1任务，不重启旧链。

空间保护0/4，主要A/B各0/28、完整资格均False。独立空间相对F0低2.29–3.98dB；学习和固定因果输出在全部20条件都被其自身DIS post三指标支配。相对自身preview仍有19/20时间改善等局部正例，详[全部成对结果](report/v1-analysis/tables.md)，不否定所有历史信息或低计算骨干。按原协议停止固定核/门的宽度、phase倍率、alpha、噪声图、loss、LR、长度/seed网格，不扩该失败候选成本/外部/手机。

[完整共同表](report/v1-joint/tables.md)、[全点CSV与PDF/PNG](report/v1-plots/)、[联系图](report/v1-contacts/)、[实际图表审阅](report/v1-plots/actual-review001.json)均保留；重复DEV/合成来源/2seed不构成独立视频或总体显著性证据。静态图不是全视频/MOS，也不证明混叠原因。没有可投稿方法、手机速度或独立最终结论，整篇论文目标仍active。

后续独立研究须先保住原有增强能力，明确表示/初始化转移/训练预算与直接先例，再检验时域机制。原F0继承官方LoL→R41 4K→R44 4K；新学生2K KD＋4K GT不等于同样预训练历史。当前负结果不能单独归因为packing/蒸馏/欠训练，也不能用延长同一学生或改门/phase/loss来救分。原MobileIE stride1无下采样，普通packing/KD/门/递归已有先例，不构成已确立创新。

复核来源：[原模型与训练审计](original-model-and-training.md)、[先例与原候选](precedents-and-candidate.md)、[原协议](protocol-v1.md)、[CPU执行修订](cpu-backend-amendment001.md)、[GPU组执行修订002](gpu-group-amendment002.md)、[完整过程](process/v1/journal.md)。空间16180/学习16365参数，较低解析MAC不是速度，不能叫4K或零推理开销。全部原34＋5＋CPU4/执行001的2源/002的4源保持，不改写历史失败。

按用户要求，长等待依据实际初段速度估算sleep，覆盖大部分预计余量，醒后核同PID/start/runtime与自动接续；到期不作重启或算法截止。main/自有data/指定作者/no-ff/小步提交和阶段推送，所有旧STOP/独立最终封存及survey/pre/demo/org边界保持。
