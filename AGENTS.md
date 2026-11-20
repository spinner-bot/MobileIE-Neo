# MobileIE-Neo 服务器工作约定

整理：2026-11-20T15:24:58.353622+08:00；Codex  (sqs_A100)。原因：将当前接续与长期约定置于入口，完整旧记录移至[原文快照](docs/交接历史/AGENTS-20261120-1524.md)，避免旧PID/“当前最新”提示反复进入工作上下文。本次只整理文档，不改变任何模型、协议、止损或运行任务。

## 当前工作

目录编号约束（2026-11-17T19:15:22.552814+08:00；用户明确通知）：`research/` 的7号已由其他分支占用，即使当前main工作区尚无该目录，也不得将其视为可用。新建独立研究主题目录应跳到 `research/8.*`，保留7号给原占用分支。

当前最新0173/主题8-V1 progress016：seed0已正常完成全部14000训练及110普通质量/4928逐帧（CUDA=False），完整原结果已分别提交。seed1 warm2000、spatial4000、fixed4000唯一最终均已完成/重载有限；fixed于15:37:03结束，原worker继续最后learned4000，progress016核当时learned121真实前缀，不按loss或先seed质量选模型。原CPU协调3024380/start109481893及独立seed1训练3057710/start109820958仍同身份live，各2线程/nice5/空CUDA mask；下一先核这两个/proc/start与原日志/runtime，勿由旧记录或观察超时重启。原quality3057709已正常结束、Z等待父回收，不再当live或重跑。原34科学质量＋5辅助＋CPU4执行源保持、活跃时勿改；两seed28000、全220质量/官方JOD/280共同表/四图实阅与七强参照/空间保护/成本止损不变。原监督器两seed及CPU质量完成后才调用post chain003，当前勿手工启动或恢复旧GPU链。plot_v1只在完整共同表后作全14输出图，当前未执行。尚无完整两seed/JOD、创新、手机或独立最终结论，长期goal active。读[0173](docs/进展报告/0173-T-26112015-首seed完整训练与双路自动接续.md)、[CPU修订](research/8.video-backbone/cpu-backend-amendment001.md)、[完整过程](research/8.video-backbone/process/v1/journal.md)。

核进程须匹配/proc的PID和start_ticks，再看状态及原runtime。Z表示数值子进程已结束但尚未被父进程回收，不能视为仍在训练。静态记录、旧日志、观察超时或一次观察断言都不能作为重启依据。后续只更新本段当前摘要，阶段历史写入研究journal与进展报告。

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
- 共享GPU按[资源规范](specs/项目规范/GPU资源使用规范.md)及当前冻结协议确认余量，允许经评估后与他人同卡并行；不改变他人的进程、功率或任务设置。当前V1已改为显式CPU执行，不能自行放宽原GPU门槛或重复启动GPU训练。

## 研究和交接边界

- 目标仍是有可验证创新及画质—时序—完整效率收益的论文；资格检查、正常训练、负结果或工程组合均不能替代方法成功。保留公开预训练来源、同预算差异、未来帧、反复使用DEV和真实域限制，不将解析MAC、静态图或代理指标当手机速度/视频感知/独立最终证据。
- 所有既有STOP继续有效，包括R28/R34/R37/R44–R46/R48/R50–R51/R61/R64–R65/R73–R75/R77/R79–R82/R84–R88等的局部网格；本主题也须按自身协议止损。旧决策原文在[研究决策目录](research/4.temporal-consistency/docs/)，完整历史在[原文快照](docs/交接历史/AGENTS-20261120-1524.md)。不因当前入口缩短而恢复任何已停止实验。
- 原始MobileIE/F0实际4075参数；主题8独立空间16180、学习融合16365，使用两遍骨干和DIS，不能写成4K或零额外推理开销。packing、KD、门控、递归等已知组件不是已确立创新。
- OLiVES 43/71/83、DAVIS testdev、Kodak 13–24独立最终继续封存；未经预定开发/冻结门槛，不读取最终像素或将开发集结论改称独立测试。主论文原84质量值/snapshot003/PDF020保持，当前仍非投稿完成稿。
- main只推进核心研究，survey/pre与demo/org各自worktree不修改、7号继续预留。先读[最新0173](docs/进展报告/0173-T-26112015-首seed完整训练与双路自动接续.md)、[主题8入口](research/8.video-backbone/README.md)、[原完整过程](research/8.video-backbone/process/v1/journal.md)与[论文状态](paper/RESEARCH_STATUS.md)；需要早期细节再定向查历史，勿按旧活跃行重启。
