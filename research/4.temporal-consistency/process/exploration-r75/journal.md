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

## 后续7K里程碑与解释性控制

本轮恢复后重新核原训练4160205/4160204与协调4163819真实PID/start，原7小时连续训练仍在执行；17/27源保持，不根据较早600步快照重启。1K..7K14份不可变marker检查seed/预算/owned路径/文件大小后逐文件提交，不重跑模型或Adam轨迹。progress002归档时两seed均7600，原日志/窗口只复制完整行前缀，所有live原件仍等终态完整归档。

实际代码的history槽分布差异促成R76固定self4解释控制，原模型、优化、数据和八主门槛未改变。R76仅CPU toy合同及新最终CPU读出代码已完成，没有真实模型推理或新增GPU；其JOD/汇总/固定图/补充协调仍待实现，不能把原最终协调器视作已包含它。R-DID获取403后的边界独立记在prior-art process，不扩大基准数或循环请求。0116报告与入口分别提交，paper018、旧STOP与目标active保持。

本阶段仍只fetch origin main/查分歧/检查自有diff/阶段SSH推送。所有源码改动已提交，运行中parent/progress/training/window/waiting是刻意保留的活态，不把它们伪装成已结束记录；新快照以progress002为准，后续仍必须读真实进程和runtime。

## 22:31后9000里程碑（progress004）

前序progress003及R76过程已记录各8000marker、8100/8000原训练和补充协调39049实际等待；R77另只做有条件协议/人工CPU资格。上述旧“待实现”段为历史，R76现有37源冻结及等待身份以0117/0119和本次真实观察为准，不能重复启动原任务。

本回合前三次观察均为原句柄真实存活、9000尚无；每次只等待，不把观察窗口结束当任务失败。22:31:21两seed均9000、两个regular-step009000.json实际存在且无runtime终态。随后核/data挂载、17/27/37冻结源保持、seed/提交步/预算/非final、资产在自有data且大小匹配，两个marker逐份提交。1K..9K现18份正规marker，不重读Tensor/重跑SGD或打开DEV。

progress004保存两训练4160205/start48844067、4160204/start48844061及原协调4163819/start48956211、补充39049/start51721359同身份活跃、新资源快照与两waiting元数据。running-prefix004只复制完整行training/window和当次progress；原持续流/父/等待文件保留活态，终态后再交。无模型评价、新GPU、训练重启/恢复或预算变化。

0121阶段报告及入口各自提交，fetch只取main/查分歧/自有diff检查/阶段推送。paper018、全部旧STOP/缺口与远期active保持；R77额外128K仍须原八门槛+self4四支持全True及两套actual review，当前没有原最终分数，禁止把CPU资格当已训练方法或新阶段已通过。

## 23:28后10000里程碑（progress005）

上一回合为verified wait。本回合先核原两训练和两协调真实PID/start、progress/runtime及实际正规marker文件名，不把进度声明代替落盘。23:27:57观察seed1已10000/实际regular-step010000，seed0仍9900；先核/data、自有资产大小/提交步/预算/非final后独立提交seed1。23:28:19再核seed0实际10000及marker，另作一份提交，不回写成同时完成。

随后核原17训练源/协议、27及37评价联合源保持；两原训练4160205/4160204与原协调4163819、补充39049均同身份活跃，无runtime终态。progress005另存实际资源/waiting快照；running-prefix005仅复制已解析完整行training/window与当次progress，不把原活态文件当完整终态。1K..10K共20份正规marker，Tensor检查点仍仅自有data，不重读Tensor/重放SGD或打开中途DEV。

没有新模型/预算变化/训练恢复重启/质量/JOD或方法结论。R75完整64000→原完整评价→R76补充及实际图审边界保持；R77仍条件登记/CPU资格，保留OLiVES仍只metadata、不因原门槛通过自动解封。0123报告及接续入口分开小步提交，阶段fetch仅main/实际分歧no-ff/自有diff/推送；paper018、旧STOP/全部论文缺口和远期active保持。

## 11月14日11000里程碑（progress006）

前序回合均为原PID/start真实存活后的verified wait。00:25:55先观察seed1实际11000及regular-step011000落盘、seed0仍10900；确认/data挂载、seed/提交步/下一步/64000预算/非final、自有data资产路径和大小后单独提交seed1元数据。00:26:21再核seed0实际11000及marker，另交seed0，不将两路写成同时完成。

00:27:11的progress006核两训练4160205/start48844067、4160204/start48844061和原协调4163819/start48956211、补充39049/start51721359同身份活跃，无runtime终态；17训练源/协议及27/37评价联合源保持。running-prefix006按seed分别保存已解析完整行training/window及当次progress，原持续流仍等终态。1K..11K周期marker现22份，另有首步marker两份，实际regular文件共24；大体量Tensor仍只存自有data，未加载Tensor、重放SGD或做中途DEV。

首次过程追加补丁误用了报告结尾作上下文，校验失败且无文件变更，随后按真实journal末段追加。无训练/评价源码修改、重启、恢复、额外预算或新质量/JOD。0124报告和入口逐文件另交，本阶段fetch仅main、如有分歧显式no-ff，完成检查后推送。R75→R76及两套实际图审仍待；R77严格条件、全部保留像素限制、paper018、旧STOP与论文缺口保持，远期目标active。


## 01:24双种子12000里程碑（progress007）

前序回合均为原PID/start真实存活后的verified wait。01:22:38观察seed1实际12000及regular-step012000落盘、seed0仍11900；确认/data挂载、seed/提交步/下一步/64000预算/非final、自有data资产路径和大小后单独提交seed1。01:23:42再核seed0实际12000及marker，另交seed0，不写成同时完成。

progress007于01:24:18核两训练4160205/start48844067、4160204/start48844061及原协调4163819/start48956211、补充39049/start51721359同身份活跃，无runtime终态；17训练源/协议和27/37评价联合源保持。running-prefix007按seed分别保存已解析完整行training/window与当次progress，原持续流仍待终态。1K..12K周期marker共24份，另有首步2份，总26；Tensor仍只存自有data，未加载Tensor/重放SGD/做中途DEV。

无训练或评价源码改动、重启、恢复、预算变化、额外训练或最终质量/JOD。0125报告与入口逐文件另交，本阶段仅fetch main、必要时显式no-ff合并、检查后推送。R75→R76完整评价与两套actual review仍待；R77严格条件、保留像素保护、paper018、旧STOP、全部论文缺口和目标active保持。


## 02:21双种子13000里程碑（progress008）

前序为原PID/start核活后的verified wait。02:20:02观察seed1实际13000及regular-step013000落盘、seed0仍12900；确认/data挂载、seed/提交步/下一步/64000预算/非final、自有data路径和资产大小后单独提交seed1。02:21:10再核seed0实际13000及marker后另交，不写成同时完成。

progress008于02:21:43核两训练4160205/start48844067、4160204/start48844061及原协调4163819/start48956211、补充39049/start51721359同身份活跃，无runtime终态；17训练源/协议和27/37评价联合源保持。running-prefix008按seed保存已解析完整行training/window及当次progress，原持续流待终态。1K..13K周期marker共26份，另首步2份；Tensor仅自有data，未加载Tensor/重放SGD/做中途DEV。

无源码改动、重启、恢复、改预算、额外训练或最终质量/JOD。0126报告和入口逐文件另交，fetch仅main、必要时显式no-ff合并、检查后推送。R75→R76完整评价及两套actual review仍待；R77严格条件、保留像素保护、paper018、旧STOP、全部论文缺口与目标active保持。


## 03:19双种子14000里程碑（progress009）

前序为原PID/start核活后的verified wait。03:17:39观察seed1实际14000及regular-step014000落盘、seed0仍13900；确认/data挂载、seed/提交步/下一步/64000预算/非final、自有data路径和资产大小后独立提交seed1。03:18:49再核seed0仍无新marker，继续等同一活跃句柄；03:19:19观察其实际14000及marker后另交，不写成同时完成。

progress009于03:19:52核两训练4160205/start48844067、4160204/start48844061及原协调4163819/start48956211、补充39049/start51721359同身份活跃，无runtime终态；17训练源/协议与27/37评价联合源保持。running-prefix009按seed保存已解析完整行training/window及当次progress，原持续流待终态。1K..14K周期marker共28份，另首步2份；Tensor仅自有data，未加载Tensor/重放SGD/做中途DEV。

无源码改动、重启、恢复、改预算、额外训练或最终质量/JOD。0127报告和入口逐文件另交，fetch仅main、必要时显式no-ff合并、检查后推送。R75→R76完整评价及两套actual review仍待；R77严格条件、保留像素保护、paper018、旧STOP、全部论文缺口与目标active保持。

## 04:17双种子15000里程碑（progress010）

前序为原PID/start核活后的verified wait。本轮持续核查同一组真实句柄，04:15:25观察seed1实际15000及regular-step015000落盘、seed0仍14900；确认/data挂载、seed/提交步/下一步/64000预算/非final、自有data路径和资产大小后独立提交seed1。04:16:29再核seed0仍无新marker，继续等待；04:16:57观察其实际15000及marker后另交，不写成同时完成。

progress010于04:17:32核两训练4160205/start48844067、4160204/start48844061及原协调4163819/start48956211、补充39049/start51721359同身份活跃，无runtime终态；17训练源/协议与27/37评价联合源保持。running-prefix010按seed保存已解析完整行training/window及当次progress，各151条training、16条window；原持续流待终态。1K..15K周期marker共30份，另首步2份；Tensor仅自有data，未加载Tensor/重放SGD/做中途DEV。

无源码改动、重启、恢复、改预算、额外训练或最终质量/JOD。0128报告和入口逐文件另交，fetch仅main、必要时显式no-ff合并、检查后阶段推送。R75→R76完整评价及两套actual review仍待；R77严格条件、保留像素保护、paper018、旧STOP、全部论文缺口与目标active保持。

## 05:15双种子16000里程碑（progress011）

前序为真实原PID/start核活后的verified wait。本轮持续核查，05:13:05观察seed1实际16000及regular-step016000落盘、seed0仍15900；确认/data、seed/提交步/下一步/64000预算/非final、自有资产路径和大小后独立提交seed1。05:14:05再核seed0仍无新marker，继续等待同一进程；05:14:38观察实际16000及marker后另交，不写成同时完成。

progress011于05:15:11核原4160205/start48844067、4160204/start48844061、原协调4163819/start48956211及补充39049/start51721359同身份活跃，无runtime终态；17训练源/协议及27/37评价联合源保持。running-prefix011按seed另存完整可解析行，各161条training/17条window及当次progress；原活态流待终态。周期1K..16K共32marker，另首步2份；Tensor仅自有data，无加载Tensor、重放SGD或中途DEV。

两路各完成完整预算的四分之一，尚无final或质量/JOD/方法结论。无源码改动、重启、恢复、改预算或额外训练。0129和入口另交、仅fetch main/分歧显式no-ff、检查后阶段推送。原完整R75→R76及两套实际图审、R77严格条件、保留像素保护、paper018、旧STOP、全部论文缺口与目标active保持。

## 06:12双种子17000里程碑（progress012）

前序为真实原PID/start核活后的verified wait。本轮持续核查，06:10:43观察seed1实际17000及regular-step017000落盘、seed0仍16900；确认/data、seed/提交步/下一步/64000预算/非final、自有资产路径和大小后独立提交seed1。06:11:39再核seed0仍无新marker，继续等待同一进程；06:12:09观察实际17000及marker后另交，不写成同时完成。

progress012于06:12:41核原4160205/start48844067、4160204/start48844061、原协调4163819/start48956211及补充39049/start51721359同身份活跃，无runtime终态；17训练源/协议及27/37评价联合源保持。running-prefix012按seed另存完整可解析行，各171条training/18条window及当次progress；原活态流待终态。周期1K..17K共34marker，另首步2份；Tensor仅自有data，无加载Tensor、重放SGD或中途DEV。

尚无final或质量/JOD/方法结论，无源码改动、重启、恢复、改预算或额外训练。0130和入口另交、仅fetch main/分歧显式no-ff、检查后阶段推送。原完整R75→R76及两套实际图审、R77严格条件、保留像素保护、paper018、旧STOP、全部论文缺口与目标active保持。

## 07:09双种子18000里程碑（progress013）

前序为真实原PID/start核活后的verified wait。本轮持续核查，07:08:23观察seed1实际18000及regular-step018000落盘、seed0仍17900；确认/data、seed/提交步/下一步/64000预算/非final、自有资产路径和大小后独立提交seed1。继续等待同一seed0原进程，07:09:19观察实际18000及marker后另交，不写成同时完成。

progress013于07:09:52核原4160205/start48844067、4160204/start48844061、原协调4163819/start48956211及补充39049/start51721359同身份活跃，无runtime终态；17训练源/协议及27/37评价联合源保持。running-prefix013按seed另存完整可解析行，各181条training/19条window及当次progress；原活态流待终态。周期1K..18K共36marker，另首步2份；Tensor仅自有data，无加载Tensor、重放SGD或中途DEV。

尚无final或质量/JOD/方法结论，无源码改动、重启、恢复、改预算或额外训练。0131和入口另交、仅fetch main/分歧显式no-ff、检查后阶段推送。原完整R75→R76及两套实际图审、R77严格条件、保留像素保护、paper018、旧STOP、全部论文缺口与目标active保持。
