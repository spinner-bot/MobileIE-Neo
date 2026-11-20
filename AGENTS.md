# MobileIE-Neo 服务器工作约定

整理：2026-11-20T15:24:58.353622+08:00；Codex  (sqs_A100)。原因：将当前接续与长期约定置于入口，完整旧记录移至[原文快照](docs/交接历史/AGENTS-20261120-1524.md)，避免旧PID/“当前最新”提示反复进入工作上下文。本次只整理文档，不改变任何模型、协议、止损或运行任务。

## 当前工作

目录编号约束（2026-11-17T19:15:22.552814+08:00；用户明确通知）：`research/` 的7号已由其他分支占用，即使当前main工作区尚无该目录，也不得将其视为可用。新建独立研究主题目录应跳到 `research/8.*`，保留7号给原占用分支。

当前最新0186/主题8-S1单向传输贡献者离散度筛查已完整负结项：[0186](docs/进展报告/0186-T-26112022-单向传输贡献者离散度筛查与止损.md)、[决定](research/8.video-backbone/single-direction/decision.md)、[原真实终态](research/8.video-backbone/single-direction/process/terminal001.json)、[过程](research/8.video-backbone/single-direction/process/journal.md)。双冻结4075参数F0/原两训练批、64NN/16窗/80输出/320逐帧、48双向DIS对/96calc及48共同GT-derived评分FB对，0优化/GPU/DEV/最终。原3364020/start112572637于22:39:38正常38.999856秒结束，session20567 exit0、真实/proc原句柄已无，当前没有任何S1/P1/T2/T1/V1活跃作业。三纯数组资格/16窗原pipeline缓存fields差0，26具名冻结保持。对原FB仅ND0一组通过；SYN0/1、ND1 PSNR低.51–.73dB，对单backward的5%时间门四组都失败，三个完整门False；对原FB仍9/16窗口联合正、对raw12/16联合正，不能抹掉局部收益。全20均值及全部16窗t3 PNG实际view_image，PDF只导出，非播放/MOS/手机/独立最终。停止固定目标贡献者方差/mass界/阈值/alpha/preset/field来源网格，不训练补救/扩失败DEV/成本/手机；直接splat/flow方差已有先例，未建立创新或加速。按旧同加载器速度sleep60秒后一次收齐，无墙钟截止，不按旧state重启。

先前P1原1280NN/0优化/200输出已完整负结项：[0185](docs/进展报告/0185-T-26112022-粗能量历史零更新筛查与止损.md)、[P1决定](research/8.video-backbone/phase-envelope/decision.md)。18具名冻结/80首帧差0保持，只有clean_mapping通过，固定粗能量/差分/alpha/尺度/clip/epsilon网格STOP；不据clean正例追加训练。新核OLiVES Genlock/25fps采集不能反改旧nominal30fps JOD或未经抽帧间隔核查声称缓存连续25fps。

先前T2全16000正式更新/四唯一最终/120质量与120官方JOD/360共表已负结项：[0184](docs/进展报告/0184-T-26112021-部署递归监督完整评价与止损.md)、[T2决定](research/8.video-backbone/recurrent-objective/decision.md)。恢复0/4、完整A/B与成本False，对独立clip-GT DIS全部20条件时间更高，固定原F0/native DIS/.5/T4递归监督网格继续STOP；47全链/24资格源和原协议不改，不再向T1/T2加loss/延长。V1/T1/R51/R73及所有旧STOP、原84值/snapshot003/PDF020/最终封存/7号/其他worktree保持。下一问题尚未登记，必须实质不同信息/结构及有限投资验证；不预宣布有效、不按旧state重启。4075非用户硬上限/零状态非唯一路线，已冻协议不放宽。论文goal active，尚无确立创新/真实自然夜间/手机/独立最终证据。

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
- main推进核心研究；移动演示支线不修改。Git作者使用 `Codex  (sqs_A100)`，保持文件/模块级小步提交，过程/结果/报告分别提交，阶段完成推送。
- pull/merge显式使用 `--no-ff`；不rebase、squash或强推。遵守[版本规范](specs/项目规范/版本管理规范.md)及[路径规范](specs/项目规范/路径与环境规范.md)。
- 本仓库SSH身份使用zrh自有密钥与隔离core.sshCommand，GitHub账号spinner-bot；不使用共享服务器其他账号的默认SSH配置，不记录密钥内容。详[版本规范](specs/项目规范/版本管理规范.md)。
- 共享GPU按[资源规范](specs/项目规范/GPU资源使用规范.md)及当前冻结协议确认余量，允许经评估后与他人同卡并行；不改变他人的进程、功率或任务设置。V1的CPU训练历史与GPU评分执行修订须保留，不能自行放宽原GPU门槛或重复启动已完成训练。

## 研究和交接边界

- 目标仍是有可验证创新及画质—时序—完整效率收益的论文；资格检查、正常训练、负结果或工程组合均不能替代方法成功。保留公开预训练来源、同预算差异、未来帧、反复使用DEV和真实域限制，不将解析MAC、静态图或代理指标当手机速度/视频感知/独立最终证据。
- 所有既有STOP继续有效，包括R28/R34/R37/R44–R46/R48/R50–R51/R61/R64–R65/R73–R75/R77/R79–R82/R84–R88等的局部网格；本主题也须按自身协议止损。旧决策原文在[研究决策目录](research/4.temporal-consistency/docs/)，完整历史在[原文快照](docs/交接历史/AGENTS-20261120-1524.md)。不因当前入口缩短而恢复任何已停止实验。
- 原始MobileIE/F0实际4075参数；主题8-V1独立空间16180、学习融合16365，使用两遍骨干和DIS，不能写成4K或零额外推理开销。packing、KD、门控、递归等已知组件不是已确立创新。
- OLiVES 43/71/83、DAVIS testdev、Kodak 13–24独立最终继续封存；未经预定开发/冻结门槛，不读取最终像素或将开发集结论改称独立测试。主论文原84质量值/snapshot003/PDF020保持，当前仍非投稿完成稿。
- main只推进核心研究，survey/pre与demo/org各自worktree不修改、7号继续预留。先读[最新0186](docs/进展报告/0186-T-26112022-单向传输贡献者离散度筛查与止损.md)、[主题8入口](research/8.video-backbone/README.md)、[原完整过程](research/8.video-backbone/process/v1/journal.md)与[论文状态](paper/RESEARCH_STATUS.md)；需要早期细节再定向查历史，勿按旧活跃行重启。
