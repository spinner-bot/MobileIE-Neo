# MobileIE-Neo 服务器工作约定

整理：2026-11-20T15:24:58.353622+08:00；Codex  (sqs_A100)。原因：将当前接续与长期约定置于入口，完整旧记录移至[原文快照](docs/交接历史/AGENTS-20261120-1524.md)，避免旧PID/“当前最新”提示反复进入工作上下文。本次只整理文档，不改变任何模型、协议、止损或运行任务。

## 当前工作

目录编号约束（2026-11-17T19:15:22.552814+08:00；用户明确通知）：`research/` 的7号已由其他分支占用，即使当前main工作区尚无该目录，也不得将其视为可用。新建独立研究主题目录应跳到 `research/8.*`，保留7号给原占用分支。

当前最新0176/主题8-T1 teacher-targets：V1已完整负结项（0175/terminal001），所有旧V1任务终态，固定packed核/门/phase/loss/长度网格STOP保持。新独立问题为保留同seed原4075参数F0后比较训练监督来源。R28真实监督GT而非strong-post输出KD已再次核代码/既有勘误，OFD2020/PocketDVDNet直接先例已读，普通KD/同F0残差/教师GT混合均不主张创新。teacher-targets预检先注册后冻21源，唯一CPU preflight001正常36.961052秒、旧PID3145707已退出，无活跃T1/V1/GPU任务，不重跑。原32批B4T4/128/128训练窗、512冻结F0图/384教师DIS/384 GT参考half-FB，0 SGD/学生前向/新DEV/最终/GPU；四组MSE比.9292–.9927、GT时间代理比.7937–.8260，必要前提4/4。仅训练目标信号，GT代理非物理true flow，也不证明学生可学或方法新颖。另[完整T1协议](research/8.video-backbone/teacher-targets/training-protocol.md)已登记同F0 gt/spatial/temporal三组各4000×两seed24000、全180质量/JOD/240共表/48均值/十强参照/四固定图与成本止损；当前完整trainer/evaluation/协调未实现，资格0/正式更新0。下一先完成全链代码和冻结，再CPU/CUDA各6丢弃资格及唯一完整预算，勿以预检或部分训练替代最终比较。V1原28000/全220质量/JOD/280表与全部实际阅图仍完整保存，原84表值/snapshot003/PDF020、所有旧STOP/独立最终封存/7号和其他worktree保持，论文goal active尚无新方法。长等待按速度估算sleep减少激活，不作deadline或到期重启。先读[0176](docs/进展报告/0176-T-26112017-训练目标预检与已知时序KD对照登记.md)、[T1入口](research/8.video-backbone/teacher-targets/README.md)、[预检决定](research/8.video-backbone/teacher-targets/preflight-decision.md)、[T1过程](research/8.video-backbone/teacher-targets/process/journal.md)，旧V1详[0175](docs/进展报告/0175-T-26112017-完整视频骨干候选评价与止损.md)。

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
- main只推进核心研究，survey/pre与demo/org各自worktree不修改、7号继续预留。先读[最新0176](docs/进展报告/0176-T-26112017-训练目标预检与已知时序KD对照登记.md)、[主题8入口](research/8.video-backbone/README.md)、[原完整过程](research/8.video-backbone/process/v1/journal.md)与[论文状态](paper/RESEARCH_STATUS.md)；需要早期细节再定向查历史，勿按旧活跃行重启。
