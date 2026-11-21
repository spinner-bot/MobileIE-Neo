# MobileIE-Neo 服务器工作约定

整理：2026-11-20T15:24:58.353622+08:00；Codex  (sqs_A100)。原因：将当前接续与长期约定置于入口，完整旧记录移至[原文快照](docs/交接历史/AGENTS-20261120-1524.md)，避免旧PID/“当前最新”提示反复进入工作上下文。本次只整理文档，不改变任何模型、协议、止损或运行任务。

## 当前工作

目录编号约束（2026-11-17T19:15:22.552814+08:00；用户明确通知）：`research/` 的7号已由其他分支占用，即使当前main工作区尚无该目录，也不得将其视为可用。新建独立研究主题目录应跳到 `research/8.*`，保留7号给原占用分支。

当前工作分支为 **research/O1**，由用户明确指定；目录仍为 `/data/Users/zrh/repo/MobileIE-Neo`。从main的bcc4198a55b91785a4a2473d4532c7017fe2160e原地新建分支，迁移当时未改main指针/已有历史，未搬源码/环境/数据，其他worktree不改。[0195迁移报告](docs/进展报告/0195-T-261121-切换research-O1分支与结果归档.md)、[迁移记录](research/8.video-backbone/operator-core/process/branch-transition001.json)。旧目标或冻结文档里的main为迁移前位置约定，以本条及用户最新指令为准，不因此自动切回main；未来合回仍 `--no-ff`。 推送前核实（2026-11-21T10:37:33.774251+08:00）main已由独立main-integration worktree推进至09f446e51；本分支不修改或回退它，历史迁移起点不是main永久冻结约束，见[并行记录](research/8.video-backbone/operator-core/process/concurrent-main001.json)。

当前运行[F1单步创新平滑参照](research/8.video-backbone/fixed-lag/README.md)，更新：2026-11-21T14:40:55.475023+08:00；Codex  (sqs_A100)。原19科学源不改，首次launch001只在语法解析失败/0资格像素；另3执行源冻结的纯语法修订后，原数值worker3938217/start118309304正常运行、人工资格通过。120序列/5016逐帧/24均值/16门全齐后才判读，不读部分分数。0新CNN/训练/GPU，明确一未来/一登记帧等待；经典RTS/运动补偿不是创新，反向采样非校准协方差增益。原递归F_t不被新输出反馈；对同未来/同场直接混合、原三基线和D1 center3完整比较。原runtime/同句柄为终态依据，不因旧PID/超时重跑；按原协议任一门失败即STOP固定函数及补救网格。此前所有STOP、最终封存与并行worktree保持，论文目标active未完成。

此前当前最新为[0204来源时序核查](docs/进展报告/0204-T-261121-来源帧索引与时间基准核查.md)，2026-11-21T14:23:17.431948+08:00；Codex  (sqs_A100)。[结论](research/8.video-backbone/cadence-audit/findings.md)/[时间契约](research/8.video-backbone/cadence-audit/results/audit001/time-contract.json)：76来源组/32held draw本地索引保持步长1；原SYN中30源属DAVIS2016报告24fps、30新增源fps未确立，D1 bear/lucia为原2016、walking不是；OLiVES16源作者报告25fps但源导出PTS未独立验证。三手机952旧decoder位置中Huawei/Pixel间隔变动、iPhone近33.333ms常量，不当sensor或手机增强时延。仅元数据/源码/两个split文字成员，0新像素/解码/CNN/flow/训练/GPU，旧名义30fps JOD/指标不回写。核查已完成，不无限追索缺失PTS；后续按登记帧/来源名义率及证据等级声明，下一先核固定时滞递归平滑的直接先例与统计/信息前提，尚未登记新实验。D1正常终态/完整3/16/固定配置与补救网格STOP保持（[0203](docs/进展报告/0203-T-261121-单帧等待信息筛查完整结果与止损.md)），原3914373已无/不重跑；全部旧STOP/原84/PDF020/最终封存/7号/并行worktree保持，论文目标active未完成。

此前最新报告为[0202同输入证据归并](docs/进展报告/0202-T-261121-同输入证据归并与时间信息预算问题.md)，更新：2026-11-21T13:45:14.512113+08:00；Codex  (sqs_A100)。[完整结论](research/8.video-backbone/evidence-reconciliation/findings.md)：L1/A1同32四帧窗×两seed的896配对/56统计，原192基线行五指标差0；实际13SYN/9ND训练场景，616场景项/616留一项/56等权统计完整，0新像素/CNN/flow/优化/GPU。独立历史时间收益对所有逐一留出仍不足1%，部分ND质量/支配结论有权重范围，原门及全部STOP不变。下一[一帧等待信息问题](research/8.video-backbone/evidence-reconciliation/next-question.md)仅问题及原长缓存metadata核查完成：10条件×两seed/三基线同身份，836内部中心/5016六臂逐帧为计划数；未打开数组或登记完整科学协议/实现/运行。先续固定协议，再有限执行，未来/等待明确另列，不当原创方法或零延迟优化；不重开失败模块网格。论文目标active未完成。

最近实验[A1局部仿射缓存](research/8.video-backbone/affine-cache/README.md)已完整负结项，更新：2026-11-21T13:29:13.971968+08:00；Codex  (sqs_A100)。[0201](docs/进展报告/0201-T-261121-局部仿射算子缓存筛查与止损.md)、[决定](research/8.video-backbone/affine-cache/decision.md)：64缓存窗/32唯一训练场景抽样窗×两原F0 seed，768输出/3072逐帧/48均值，0新CNN/优化/GPU；21具名源保持，人工资格通过不等于方法成功。表示0/4、强组合0/16，完整0/20；current SYN质量已低2.73–2.81dB，cached进一步退化。ND时间及相对hold正例保留。固定quarter/5×5/RGB仿射/eps1e−4/K2/无warp/目标/clip及补救网格STOP，不加warp/guide/训练/周期救分，不扩失败成本/DEV/JOD/手机；全部四图实际审阅，原3899949已无/正常终态，不重跑。下一须先归并结构与数据假设/论文机会，尚未登记新候选。论文目标active未完成，L1及所有旧STOP/原84值/PDF020/最终封存/7号/并行main和其他worktree保持。

此前L1已完整负结项，见[0200](docs/进展报告/0200-T-261121-潜在状态完整学习评价与止损.md)、[决定](research/8.video-backbone/latent-core/decision.md)、[终态](research/8.video-backbone/latent-core/process/terminal001.json)。更新：2026-11-21T12:58:24.776394+08:00；Codex  (sqs_A100)。两seed各两目标8000/共32K正式更新/512K学生、四唯一最终、同32训练场景窗/704输出/2816逐帧/44均值正常完成。空间2/8、历史0/12、强post0/24，A/B各2/44且完整False。八空间均值PSNR正、ND1近2dB及逐窗正例保留，不掩盖最差退化；独立历史时间不足1%，不构成方法。固定32ch/d4-d8/门/LN/读回/初始化/GT MSE/目标/LR/长度/追加seed网格STOP，不加flow/teacher/更长窗口救分、不扩失败成本/DEV/JOD/外部/手机，不否定全部latent/RNN。

原[协议](research/8.video-backbone/latent-core/training-protocol.md)、26 CPU/44全链＋1执行冻结保持；CPU8/CUDA4资格全丢弃，原外层143/无父runtime与正常CUDA007/恢复后全链正常分开保存。原父3852634、两driver/worker/held等全部句柄已无，无活跃L1/GPU，不按旧PID或stage重启。完整640配对/40统计、四PNG实际审阅和固定四行（两输入×两模型seed）联系图/PDF归档，PSNR上限0触发。L1之后A1已负结项，后续仍须归并空间/传输/响应/自由状态证据后提出不同信息/结构，不继续一般模块排列；论文目标active未完成，原84值/snapshot003/PDF020/独立最终/7号/并行main及其他worktree保持。12431参数/解析8.982835344G/state16588800B非完整成本/手机，普通GRU非创新。

长期研究目标已恢复 **active**，目标位置research/O1。更新：2026-11-21T10:36:18.033361+08:00；Codex  (sqs_A100)。原chain002已10:04:59自然完成且全部原句柄无，完整复核见[0196](docs/进展报告/0196-T-261121-共享核心完整学习评价与止损.md)、[决定](research/8.video-backbone/operator-core/decision.md)、[终态](research/8.video-backbone/operator-core/process/terminal001.json)。16K正式更新/四唯一最终/32训练抽样窗/416输出/52均值，全部四图实际审阅；空间8/8、历史0/8、目标2/4、强post4/16，完整14/36失败。局部正例保留，固定共享8通道/两级soft相关/温度-confidence/probe/目标/LR/长度/seed网格STOP，不换teacher/GT loss/延长补救，不开失败DEV/JOD/成本/手机。没有新模型推理；下一须归并证据后登记不同信息/结构，不重跑或按旧状态接续。0195的paused是迁移时状态。

25 CPU/39数值全链＋2执行源不变；CPU8/成功CUDA4资格更新全丢弃，原CUDA001环境失败及部分前向/0优化保持。5049参数、解析Conv8.060601744G、状态29030400B非手机/峰值RAM证据，基础共享/相关/KD/Sobolev不当创新。原84值/snapshot003/PDF020、独立最终/所有旧STOP/7号和其他worktree保持。后续O1代码/结果/过程/报告均在research/O1密集提交、阶段推送。

先前B1完整16窗/80输出/320逐帧、0优化/GPU，历史相对当前同表时间仅改善.167%–.744%，完整门失败，[0191](docs/进展报告/0191-T-26112101-外观索引特征历史完整筛查与止损.md)及其固定表/key/节点/alpha/decay/阈值/seed/训练补救STOP保持，原3450505已终态不重跑。

先前M1全32000正式更新/512000学生图/四唯一最终、1536NN/384窗口输出/12均值完成，八项恢复保护0/8，[0190](docs/进展报告/0190-T-26112101-多线性骨干完整空间恢复与止损.md)与[决定](research/8.video-backbone/multiaffine-core/spatial-decision.md)保持。固定width16/stride2/dilation2/init/目标/LR/长度/seed网格STOP，CPU19/20与全链40冻结不改；外层143与原数值正常完成分开留证，全部原句柄终态不重跑。F0有不同预训/expanded历史，M1负结果不作纯架构因果或普遍不可能结论。

O1固定配置已按原恢复/历史/强post门槛完整止损；不能仅换历史存放位置、把空间平滑当时间贡献或继续无界参数筛查。论文长期目标未完成，当前active，尚无确立方法/创新/手机/独立最终，原84值/snapshot003/PDF020保持。

先前N1完整11264NN统计校准结项：[0187](docs/进展报告/0187-T-26112023-三观测方差识别校准与骨干问题收敛.md)、[决定](research/8.video-backbone/triple-variance/decision.md)，16冻结保持、固定20/20/20为8/8/异photon4/16，固定三观测/16tile/截断/窗口/模式/相关构造STOP。N1/S1/P1/T2/T1/V1原句柄均终态，不重启；它们失败不改变M1新空间协议，也不构成M1成功。所有旧STOP及独立最终/原84值/PDF020/7号/其他worktree保持。

先前S1原64NN/80输出完整负结项：[0186](docs/进展报告/0186-T-26112022-单向传输贡献者离散度筛查与止损.md)、[S1决定](research/8.video-backbone/single-direction/decision.md)。26具名冻结/原pipeline差0保持，三个完整门False；停止固定目标贡献者方差/mass界/阈值/alpha/preset/field来源网格，不扩失败DEV/成本/训练；原全部句柄终态。

先前P1原1280NN/0优化/200输出已完整负结项：[0185](docs/进展报告/0185-T-26112022-粗能量历史零更新筛查与止损.md)、[P1决定](research/8.video-backbone/phase-envelope/decision.md)。18具名冻结/80首帧差0保持，只有clean_mapping通过，固定粗能量/差分/alpha/尺度/clip/epsilon网格STOP；不据clean正例追加训练。新核OLiVES Genlock/25fps采集不能反改旧nominal30fps JOD或未经抽帧间隔核查声称缓存连续25fps。

先前T2全16000正式更新/四唯一最终/120质量与120官方JOD/360共表已负结项：[0184](docs/进展报告/0184-T-26112021-部署递归监督完整评价与止损.md)、[T2决定](research/8.video-backbone/recurrent-objective/decision.md)。恢复0/4、完整A/B与成本False，对独立clip-GT DIS全部20条件时间更高，固定原F0/native DIS/.5/T4递归监督网格继续STOP；47全链/24资格源和原协议不改，不再向T1/T2加loss/延长。V1/T1/R51/R73及所有旧STOP、原84值/snapshot003/PDF020/最终封存/7号/其他worktree保持。下一问题尚未登记，必须实质不同信息/结构及有限投资验证；不预宣布有效、不按旧state重启。4075非用户硬上限/零状态非唯一路线，已冻协议不放宽。论文长期目标未完成，当前active，尚无确立创新/真实自然夜间/手机/独立最终证据。

核进程须匹配/proc的PID和start_ticks，再看状态及原runtime。Z表示数值子进程已结束但尚未被父进程回收，不能视为仍在训练。静态记录、旧日志、观察超时或一次观察断言都不能作为重启依据。后续只更新本段当前摘要，阶段历史写入研究journal与进展报告。 长等待按近期实际速度估算sleep，合理覆盖大部分预计余量，避免高频轮询和无效思考；用户消息可打断等待，醒后继续自主工作，不以时间到达截断算法预算。

## 路径、环境和版本管理

- 本项目属于zrh。源码仓库为 `/data/Users/zrh/repo/MobileIE-Neo`；项目运行资产统一在 `/data/Users/zrh/data/MobileIE-Neo`。
- 禁止借用其他用户的虚拟环境、site-packages、数据或缓存；禁止在 `/root`、其他用户目录或全局Python中建立/安装本项目环境。
- 独立环境目标为资产目录下 `.venv`；仓库 `.venv` 指向它。仓库 `.cache` 指向资产目录下 `cache`，数据集、下载权重、训练检查点、运行缓存均经此入口进入自有数据目录。
- Python环境不得启用system-site-packages，不得以pth/PYTHONPATH引用其他用户目录。系统Python标准库及系统GPU驱动可只读使用。
- 后续启动使用项目自有环境入口；pip、Torch、Hugging Face、Matplotlib及临时缓存均设置到自有资产目录，不依赖默认/root缓存。
- 代码、版本化报告/实验记录和源码自带的小型资源随仓库管理；大体量运行资产不提交Git，路径、来源和迁移核验记录入库。
- 资产与独立环境迁移于2026-11-12 03:06完成核验，仓库两个相对链接已入Git；后续统一使用 `./scripts/project-python`，禁止回退到旧环境。
- 写入前确认/data挂载；共享GPU启动前连续检查显存、利用率、计算进程并估算本任务峰值和既有任务增长余量，允许与他人任务同卡并行，同组最多四张卡，不以墙钟截断算法预算。
- 当前O1核心研究在research/O1推进；main由独立worktree接续，阶段集成显式--no-ff；移动演示支线不修改。Git作者使用 `Codex  (sqs_A100)`，保持文件/模块级小步提交，过程/结果/报告分别提交，阶段完成推送。
- pull/merge显式使用 `--no-ff`；不rebase、squash或强推。遵守[版本规范](specs/项目规范/版本管理规范.md)及[路径规范](specs/项目规范/路径与环境规范.md)。
- 本仓库SSH身份使用zrh自有密钥与隔离core.sshCommand，GitHub账号spinner-bot；不使用共享服务器其他账号的默认SSH配置，不记录密钥内容。详[版本规范](specs/项目规范/版本管理规范.md)。
- 共享GPU按[资源规范](specs/项目规范/GPU资源使用规范.md)及当前冻结协议确认余量，允许经评估后与他人同卡并行；不改变他人的进程、功率或任务设置。V1的CPU训练历史与GPU评分执行修订须保留，不能自行放宽原GPU门槛或重复启动已完成训练。
- 2026-11-21T07:22:11.808873+08:00用户新增授权：物理1、2号卡任务均属本人，允许本项目使用；以后无需再为使用这两卡请求许可，仍按实际余量准入。用户随后告知8卡已空余，NVML已核全0；该瞬时空闲不替代未来采样。该授权到达时O1原链已进资格013，本轮沿原4–7接续，不因新增卡许可重复启动。

## 研究和交接边界

- 目标仍是有可验证创新及画质—时序—完整效率收益的论文；资格检查、正常训练、负结果或工程组合均不能替代方法成功。保留公开预训练来源、同预算差异、未来帧、反复使用DEV和真实域限制，不将解析MAC、静态图或代理指标当手机速度/视频感知/独立最终证据。
- 所有既有STOP继续有效，包括R28/R34/R37/R44–R46/R48/R50–R51/R61/R64–R65/R73–R75/R77/R79–R82/R84–R88等的局部网格；本主题也须按自身协议止损。旧决策原文在[研究决策目录](research/4.temporal-consistency/docs/)，完整历史在[原文快照](docs/交接历史/AGENTS-20261120-1524.md)。不因当前入口缩短而恢复任何已停止实验。
- 原始MobileIE/F0实际4075参数；主题8-V1独立空间16180、学习融合16365，使用两遍骨干和DIS，不能写成4K或零额外推理开销。packing、KD、门控、递归等已知组件不是已确立创新。
- OLiVES 43/71/83、DAVIS testdev、Kodak 13–24独立最终继续封存；未经预定开发/冻结门槛，不读取最终像素或将开发集结论改称独立测试。主论文原84质量值/snapshot003/PDF020保持，当前仍非投稿完成稿。
- 当前O1只在research/O1工作，survey/pre与demo/org各自worktree不修改、7号继续预留。先读[最新0204](docs/进展报告/0204-T-261121-来源帧索引与时间基准核查.md)、[此前0203](docs/进展报告/0203-T-261121-单帧等待信息筛查完整结果与止损.md)、[此前0202](docs/进展报告/0202-T-261121-同输入证据归并与时间信息预算问题.md)、[此前0201](docs/进展报告/0201-T-261121-局部仿射算子缓存筛查与止损.md)、[此前0200](docs/进展报告/0200-T-261121-潜在状态完整学习评价与止损.md)、[主题8入口](research/8.video-backbone/README.md)、[原完整过程](research/8.video-backbone/process/v1/journal.md)与[论文状态](paper/RESEARCH_STATUS.md)；需要早期细节再定向查历史，勿按旧活跃行重启。
