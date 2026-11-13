# R75正式参考实现、资格与实际启动

2026-11-13；Codex  (sqs_A100)。前一目标回合为progress：作者原代码、原生适配及共享资源小试完成并推送，正式预算仅登记。本回合完成正式实现/资格并实际运行，不把参考适配或训练loss当方法成功。

1. 确认/data挂载、main干净/28131b54、作者/no-ff和0114接续；旧试跑4155198已退出，不重跑。正式协议64000×2、trial不继承、无DEV选中间checkpoint保持。
2. 均匀384四帧half synthetic/real采样单独提交，合法坐标从width/height−384+1抽取；不同于资源首窗口的128坐标夹紧。仅既有57/3428和12/384训练来源。更新/Adam/cosine/RNG保存与恢复、原子metadata、实际GPU执行按模块提交。
3. 每视频资源监测改为纯标准库只读NVML，CPU实际接口与NVIDIA主文档核对，初始/实际PID仍nvidia-smi交叉验证。持续busy决策隔2秒，内存前后不降频、25%合作让出；累计极值和最近16样本按100步记录，不生成百万CLI/遥测日志。协议首步前补充实现细化，不改变模型/信息/预算/门槛。
4. 先提交CPU资格脚本，再执行`CUDA_VISIBLE_DEVICES='' ./scripts/project-python research/4.temporal-consistency/scripts/check_causal_reference_r75.py --out research/4.temporal-consistency/process/exploration-r75/cpu-gate001`。PID4158710于13:32:25.945正常终态，68.533742秒；四batch覆盖两个seed首末data-step、源裁块和确定性采样、一丢弃CPU更新、恢复推理差0及下个step/LR字段通过，CUDA未初始化，0正式预算/新DEV。未扩Adam逐bit审计。原结果、runtime/终端流分别提交。
5. 用既有R32完整frames记录核3428源身份/SHA匹配及尺寸，最小480×854，全可裁384；没有为此重解码新视频。训练执行代码编译后冻结17源/协议和资格指纹，保存source-freeze001。协议头改为冻结登记说明，活跃状态另存，避免运行中修改已冻结协议。
6. `./scripts/project-python scripts/launch_causal_reference_r75.py`于13:37:09脱离会话启动两个原父4160081/4160082，分别整进程前绑定物理0/1；拓扑实际PIX/同NUMA0，最多两卡，不触碰4–7组。父原11样本准入正常；CPU真实资料/模型准备后同卡新11样本mean21.000/24.818%、空闲38925/39291MiB，实际CUDA PID4160205/4160204匹配预期。首步marker正常，源17保持。
7. 运行中不改训练源；审阅发现冻结trainer本身的resume参数可指旧marker，未执行任何恢复。另写外部受控恢复入口，强制terminal resource/同seed无活跃锁/latest marker==terminal提交步，禁止直接选旧marker重放。此修补只约束未来入口，不改已运行科学代码、不启动自动重试；最终读出还核全部attempt继承链连续到64000。
8. 新最终评价/官方CVVDP/完整八门槛/20固定图分别提交。原索引实际显示SYN六×64×480×854、ND四×16×288×512，ND是原生裁块，不是1080p。评价入口编写初按ND1080误估8192调用，执行前改为由真实shape推导4864/seed；错误版本未执行、无错误分数。执行说明单独记录，原固定10条件/接受域不改。
9. 6个评价脚本及恢复入口编译通过；原40质量/40预测引用和同index约束、最终读出缺final时阻挡通过接口检查，0新增DEV/数值评价。不是最终数值/JOD资格。随后评价联合训练27源冻结evaluation-freeze001，科学源后续活跃时不改。
10. 最终协调器4163819/start48956211于13:56:14脱离会话启动，核原两训练PID/start并每30秒记录verified live waiting，无墙钟截止、不自动训练重启。等两final后自动完整CPU质量/质量汇总/20图/共享GPU40JOD/联合汇总；20图actual review须root实际查看后另记，不能把协调器正常当人工已审阅。
11. progress001于13:59:58实际核两训练和协调PID均活跃、源17未变，snapshot两seed各300；原CUDA UUID配置/准入/首步marker均为不可变metadata，分别提交。活跃training/stdout只复制完整行前缀、parent/progress复制前缀元数据，原完整stream和终态以后独立提交。没有将原live文件当终态或复跑旧实验。所有checkpoint与环境仍仅自有data。
12. 0115报告单独提交，根/研究README和AGENTS按文件更新实际状态，保留旧STOP/paper018/目标active。没有新DEV/JOD/方法成功、自然动态、手机或独立最终结论；普通ReBotNet-XS64K自启动/分块适配与原500K/GT初始化/旧MobileIE8K像素预算差异保留。
13. 阶段fetch只取origin main，查分歧；必要时显式no-ff合并，禁止rebase/squash/强推。自有diff空白检查排除原终端流和原样vendor上游空白。阶段SSH推送后仍核真实训练/协调句柄；活跃parent/progress/training/window/waiting留在运行态，已核验前缀入库，终态再归档完整流，不为工作树表面干净隐藏或伪造它们。
