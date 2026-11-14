# 4. 时序一致性方法（Temporal Consistency）

目标：以论文为最终产出，深入核查 MobileIE 的架构与训练机制，在移动端计算预算内设计视频增强模型，用传统方法、后处理和已有视频模型对照验证画质、时间稳定性与效率收益。

最新里程碑（[0139](../../docs/进展报告/0139-T-26111416-正式训练双种子两万七千步里程碑归档.md)/[progress021](process/exploration-r75/progress021.json)）：16:01:52先核seed1实际27000、16:06:05再核seed0实际27000，两marker分别提交；各目标64000，周期1K..27K共54份，另首步2份。完整行前缀按seed另交，原live待终态。四原PID/start活跃、17/27/37源保持，无重启/恢复/预算调度变化/额外训练/中途DEV或最终JOD。原R75→R76完整评价与两套actual review仍待；R77严格条件、保留像素/paper018/旧STOP/论文缺口和目标active保持；以下旧进度仅为历史。

最新里程碑（[0138](../../docs/进展报告/0138-T-26111415-正式训练双种子两万六千步里程碑归档.md)/[progress020](process/exploration-r75/progress020.json)）：14:57:54先核seed1实际26000、15:01:49再核seed0实际26000，两marker分别提交；各目标64000，周期1K..26K共52份，另首步2份。完整行前缀按seed另交，原live待终态。四原PID/start活跃、17/27/37源保持，无重启/恢复/预算调度变化/额外训练/中途DEV或最终JOD。原R75→R76完整评价与两套actual review仍待；R77严格条件、保留像素/paper018/旧STOP/论文缺口和目标active保持；以下旧进度仅为历史。

最新里程碑（[0137](../../docs/进展报告/0137-T-26111413-正式训练双种子两万五千步里程碑归档.md)/[progress019](process/exploration-r75/progress019.json)）：13:52:41先核seed1实际25000、13:55:24再核seed0实际25000，两marker分别提交；各目标64000，周期1K..25K共50份，另首步2份。完整行前缀按seed另交，原live待终态。四原PID/start活跃、17/27/37源保持，无重启/恢复/预算调度变化/额外训练/中途DEV或最终JOD。原R75→R76完整评价与两套actual review仍待；R77严格条件、保留像素/paper018/旧STOP/论文缺口和目标active保持；以下旧进度仅为历史。

最新里程碑（[0136](../../docs/进展报告/0136-T-26111412-正式训练双种子两万四千步里程碑归档.md)/[progress018](process/exploration-r75/progress018.json)）：12:55:33先核seed1实际24000、12:57:23再核seed0实际24000，两marker分别提交；各目标64000，周期1K..24K共48份，另首步2份。完整行前缀按seed另交，原live待终态。四原PID/start活跃、17/27/37源保持，无重启/恢复/预算调度变化/额外训练/中途DEV或最终JOD。原R75→R76完整评价与两套actual review仍待；R77严格条件、保留像素/paper018/旧STOP/论文缺口和目标active保持；以下旧进度仅为历史。

最新里程碑（[0135](../../docs/进展报告/0135-T-26111411-正式训练双种子两万三千步里程碑归档.md)/[progress017](process/exploration-r75/progress017.json)）：原两seed均23000/64000，两份可恢复marker分别提交；周期1K..23K共46份，另首步2份。完整行前缀按seed另交，原活态流待终态。四原PID/start活跃、17/27/37源保持，无重启/恢复/改预算/额外训练/中途DEV或最终JOD。R75→R76完整评价与两套actual review仍待；R77严格条件、保留像素/paper018/旧STOP/论文缺口保持，目标active；以下旧进度仅为历史。

最新接续归档（[0134](../../docs/进展报告/0134-T-26111411-中断观察后原训练接续与两万二千步归档.md)/[progress016](process/exploration-r75/progress016.json)）：观察中断后11:24:57重核原四PID/start仍live，原两seed最近22400/64000、最新检查点22000。恢复监测后逐份提交21K/22K四marker，不伪称在写入时驻守；周期1K..22K共44份，另首步2份。完整行前缀按seed另交，17/27/37源保持，无重启/恢复/改预算/调度或额外训练，无中途DEV/最终JOD。原R75→R76完整评价与两套actual review、R77严格条件/保留像素/paper018/旧STOP/论文缺口保持，目标active；以下旧进度仅为历史。

最新里程碑（[0133](../../docs/进展报告/0133-T-26111409-正式训练双种子两万步里程碑归档.md)/[progress015](process/exploration-r75/progress015.json)）：原两seed均20000/64000，两份可恢复marker分别提交；周期1K..20K共40份，另首步2份。完整行前缀按seed另交，原活态流待终态。四原PID/start活跃、17/27/37源保持，无重启/恢复/改预算/额外训练/中途DEV或最终JOD。R75→R76完整评价与两套actual review仍待；R77严格条件、保留像素/paper018/旧STOP/论文缺口保持，目标active；以下旧进度仅为历史。

最新里程碑（[0132](../../docs/进展报告/0132-T-26111408-正式训练双种子一万九千步里程碑归档.md)/[progress014](process/exploration-r75/progress014.json)）：原两seed均19000/64000，两份可恢复marker分别提交；周期1K..19K共38份，另首步2份。完整行前缀按seed另交，原活态流待终态。四原PID/start活跃、17/27/37源保持，无重启/恢复/改预算/额外训练/中途DEV或最终JOD。R75→R76完整评价与两套actual review仍待；R77严格条件、保留像素/paper018/旧STOP/论文缺口保持，目标active；以下旧进度仅为历史。

最新里程碑（[0131](../../docs/进展报告/0131-T-26111407-正式训练双种子一万八千步里程碑归档.md)/[progress013](process/exploration-r75/progress013.json)）：原两seed均18000/64000，两份可恢复marker分别提交；周期1K..18K共36份，另首步2份。完整行前缀按seed另交，原活态流待终态。四原PID/start活跃、17/27/37源保持，无重启/恢复/改预算/额外训练/中途DEV或最终JOD。R75→R76完整评价与两套actual review仍待；R77严格条件、保留像素/paper018/旧STOP/论文缺口保持，目标active；以下旧进度仅为历史。

最新里程碑（[0130](../../docs/进展报告/0130-T-26111406-正式训练双种子一万七千步里程碑归档.md)/[progress012](process/exploration-r75/progress012.json)）：原两seed均17000/64000，两份可恢复marker分别提交；周期1K..17K共34份，另首步2份。完整行前缀按seed另交，原活态流待终态。四原PID/start活跃、17/27/37源保持，无重启/恢复/改预算/额外训练/中途DEV或最终JOD。R75→R76完整评价与两套actual review仍待；R77严格条件、保留像素/paper018/旧STOP/论文缺口保持，目标active；以下旧进度仅为历史。

最新里程碑（[0129](../../docs/进展报告/0129-T-26111405-正式训练双种子一万六千步里程碑归档.md)/[progress011](process/exploration-r75/progress011.json)）：原两seed均16000/64000、各预算四分之一，两份可恢复marker分别提交；周期1K..16K共32份，另首步2份。完整行前缀按seed另交，原活态流待终态。四原PID/start活跃、17/27/37源保持，无重启/恢复/改预算/额外训练/中途DEV或最终JOD。R75→R76完整评价与两套actual review仍待；R77严格条件、保留像素/paper018/旧STOP/论文缺口保持，目标active；以下旧进度仅为历史。

最新里程碑（[0128](../../docs/进展报告/0128-T-26111404-正式训练双种子一万五千步里程碑归档.md)/[progress010](process/exploration-r75/progress010.json)）：原两seed均15000/64000、两份可恢复marker分别提交；周期1K..15K共30份，另首步2份。完整行前缀按seed另交，原活态流待终态。四原PID/start活跃、17/27/37源保持，无重启/恢复/改预算/额外训练/中途DEV或最终JOD。R75→R76完整评价与两套actual review仍待；R77严格条件、保留像素/paper018/旧STOP/论文缺口保持，目标active；以下旧进度仅为历史。

最新里程碑（[0127](../../docs/进展报告/0127-T-26111403-正式训练双种子一万四千步里程碑归档.md)/[progress009](process/exploration-r75/progress009.json)）：原两seed均14000/64000、两份可恢复marker分别提交；周期1K..14K共28份，另首步2份。完整行前缀按seed另交，原活态流待终态。四原PID/start活跃、17/27/37源保持，无重启/恢复/改预算/额外训练/中途DEV或最终JOD。R75→R76完整评价与两套actual review仍待；R77严格条件、保留像素/paper018/旧STOP/论文缺口保持，目标active；以下旧进度仅为历史。

最新里程碑（[0126](../../docs/进展报告/0126-T-26111402-正式训练双种子一万三千步里程碑归档.md)/[progress008](process/exploration-r75/progress008.json)）：原两seed均13000/64000、两份可恢复marker分别提交；周期1K..13K共26份，另首步2份。完整行前缀按seed另交，原活态流待终态。四原PID/start活跃、17/27/37源保持，无重启/恢复/改预算/额外训练/中途DEV或最终JOD。R75→R76完整评价与两套actual review仍待；R77严格条件、保留像素/paper018/旧STOP/论文缺口保持，目标active；以下旧进度仅为历史。

最新里程碑（[0125](../../docs/进展报告/0125-T-26111401-正式训练双种子一万二千步里程碑归档.md)/[progress007](process/exploration-r75/progress007.json)）：原两seed均12000/64000、两份可恢复marker分别提交；周期1K..12K共24份，另有首步2份。完整行前缀按seed另交，原活态流待终态。四原PID/start活跃、17/27/37源保持，无重启/恢复/改预算/额外训练/中途DEV或最终JOD。R75→R76完整评价与两套actual review仍待，R77严格条件、保留像素/paper018/旧STOP/论文缺口保持，目标active；以下旧进度仅为历史。

最新里程碑（[0124](../../docs/进展报告/0124-T-26111400-正式训练双种子一万一千步里程碑归档.md)/[progress006](process/exploration-r75/progress006.json)）：原两seed均11000/64000、两份可恢复marker分别提交；周期1K..11K共22份，另有首步两份。完整行前缀按seed另交，原活态流待终态。四原PID/start活跃、17/27/37源保持，无重启/恢复/预算变化/中途DEV或最终JOD。R75→R76完整评价与两套实际图审仍待，R77严格条件、保留像素/paper018/旧STOP/论文缺口保持，目标active；以下较早进度仅是历史，不能据此重开作业。

最新里程碑（[0123](../../docs/进展报告/0123-T-26111323-正式训练双种子一万步里程碑归档.md)/[progress005](process/exploration-r75/progress005.json)）：原两seed均10000/64000、各10000可恢复marker分别提交，1K..10K共20正规marker；日志只归档完整行前缀。四原PID/start同身份活跃，17/27/37源保持，无重启/恢复/预算变化/新训练或最终DEV/JOD。R75→R76完整数值/两套actual review仍待；R77严格条件投资与原保留OLiVES纯metadata限制保持，paper018/旧STOP/论文全部缺口及目标active不改。以下旧进度均为历史，后续先核真实PID/runtime。

最新资料准备（[0122](../../docs/进展报告/0122-T-26111323-保留实拍来源纯元数据资格与最终像素保护.md)）：原两seed最近9700/64000、四原句柄活跃、17/27/37源保持。[既有保留OLiVES](data/external-olives-reserved/README.md)仅目录元数据资格：固定版本8请求/383301字节、0图像；实际31train+9test有双侧目录，余3保留test的200/225/161共同帧名对既有SHA无精确重合。它们仍未认证独立场景/预训练未见或下载可用，不增加已完成基准数；所有保留像素继续禁止作DEV，原门槛通过亦不自动解封。完整最终方法/权重/预处理/参考/评价及读取范围须另冻结，首16元数据指针非正式长度选择。原训练/评价继续，paper018/旧STOP/目标active保持。

最新里程碑（[0121](../../docs/进展报告/0121-T-26111322-正式训练双种子九千步里程碑归档.md)/[progress004](process/exploration-r75/progress004.json)）：原两seed均9000/64000、各9000marker及完整行前缀已分别提交，1K..9K共18正规marker，原四PID/start同身份仍活跃。17/27/37冻结源保持，无重启/恢复/中途DEV/JOD或新方法。R75→R76固定完整评价和实际图审仍待，R77仍严格条件登记/人工CPU资格，无新增正式训练/自动启动器。paper018、全部旧STOP/自然动态手机最终缺口保持，目标active；以下旧进度仅作历史，不据此重启原任务。

最新有条件准备（[0119](../../docs/进展报告/0119-T-26111322-同容量空间训练控制登记与CPU资格.md)）：R75两seed最近8400/64000、原两训练/两协调同身份活跃，17/27/37源保持。另登记[R77独立空间训练控制](docs/exploration-r77-trained-spatial-control-protocol.md)，补same-weight current/self4不能代替同容量独立训练的归因边界；原XS/数据seed-step/预算不变、每帧F(x,x)。CPU人工输入资格6.399秒/一次丢弃完整更新通过，0真实视频/正式更新/GPU。**仅原八门槛与self4四支持全部True及两套20图实际审阅后，才实施额外训练/GPU资格；当前pending、无自动启动器、完整trainer/evaluator未实现。** 有False停止额外128K投资，不扩长度/seed网格。paper018、旧STOP及全部论文缺口保持，目标active。

当前最新（[0117](../../docs/进展报告/0117-T-26111321-自条件补充评价链冻结与实际等待接续.md)/[progress003](process/exploration-r75/progress003.json)）：R75原两seed最近归档8100/8000，各64000目标，1K..8K16份marker分别提交，训练/协调真实存活，17/27源不变。尚无最终DEV/JOD。[R76同帧self4解释控制](docs/exploration-r76-self-conditioning-control-protocol.md)已补齐CPU/官方感知/100条件汇总/20追加固定图，编译/metadata屏障资格通过、37源冻结；实际补充协调39049/start51721359仅等待原4163819及完整原结果，无新增GPU/训练/真实DEV读出。固定四次完整拼接/更多计算/0过去观测，补充而不改原八门槛；实际数值、四支持检查与图审仍待。下一按真实终态接续，不改活跃37源、不重跑原训练。note58 R-DID403获取未完成；旧STOP及paper018保持，目标active。

首次正式启动归档（[0115](../../docs/进展报告/0115-T-26111314-原生因果参考正式双种子启动与最终评价链接入.md)/[progress001](process/exploration-r75/progress001.json)）：R75已正式运行，seed0 PID4160205/实际0、seed1 PID4160204/实际1，各64000，当时归档各300步。CPU原生数据/更新恢复资格和17源冻结完成；源码/完整行前缀/固定准入等分别入库，无新DEV/JOD或方法成功。协调PID4163819按真实PID/start等待两final，再跑完整40新质量/40JOD/80共同条件/20图，手工实际图审仍待执行。17训练源/协议及评价联合27源活跃时保持，不重启原训练。SYN480×854、ND288×512原生裁块，非整幅1080p，每seed最终4864 tile调用/896帧，见[执行边界](docs/exploration-r75-evaluation-execution.md)。未来真实resource退出只用latest验证入口恢复，不选旧marker重放或自动重试科学失败；当前进度以上述0116及真实runtime为准。

前阶段（[0114](../../docs/进展报告/0114-T-26111313-原生因果参考共享GPU试跑完成与正式预算登记.md)）：原生ReBotNet分块CPU资格和两次丢弃共享GPU更新均正常，实际0号匹配/own最高1242MiB/最少空闲37677MiB，PID4155198已退出，0正式更新/检查点，不重跑资源小试。当时仅登记[R75正式协议](docs/exploration-r75-causal-reference-protocol.md)，后续实现/CPU资格与实际正式训练以上述0115为准；普通参考适配不是创新/充分原表复现。

前一代码资格阶段：[0113](../../docs/进展报告/0113-T-26111313-公开因果视频参考代码资格与成本边界.md)与[note57](../5.prior-art-survey/docs/57-causal-reference-code-feasibility.md)完成公开因果RGB代码资格。ReBotNet作者原架构/CC0/配置已固定，CPU正常、890275参数、384调用Conv/Linear .784862208GMAC，阶段内0SGD/质量/新视频像素/GPU。固定576token、原文GT启动/500K训练与未提供完整trainer的边界均保留；功能资格不等于充分复现。之后的资源实测与正式登记以上述0114为准。

当前接续（2026-11-13 12:46 CST）：R74完整收口，16000正式motion、80新CPU质量/3584逐帧、80感知/3584比较帧、120条件24均值及20固定图actual review全正常，16门槛全部False，cost未运行。按[决定](docs/exploration-r74-decision.md)停止本组合局部网格，不再加教师/权重/宽度/长度等挽救。先读[0112报告](../../docs/进展报告/0112-T-26111312-运动代理监督完整止损与共享设备绑定修复.md)、[progress007](process/exploration-r74/progress007.json)和[过程](process/exploration-r74/journal.md)。所有原训练/CPU/renderer/指标/监督PID已退出，原控制器纯资源等待False终态与实际选卡错误保留；不能按旧提示重启。原科学42源未变，指标同PID恢复不重算。后续[进程前绑定入口](docs/prebound-shared-gpu-entry.md)已修复并通过真实CPU metric后物理1号的一字节零评分验证，不重跑旧80。当前无新候选登记，下一仅定向核查一个公开因果参考的实际可用性/信息/成本条件，所有旧STOP保持；paper仍build018内部证据稿，创新/自然动态/手机/最终未完成，远期目标active。

- 当前性质：前期机制预实验，尚无已建立的新方法、联合收益或最终测试。双矩/递归融合已有先例，[note54](../5.prior-art-survey/docs/54-moment-history-equivalence-and-noise-boundaries.md)明确混合离散度不等于噪声方差；不从loss或中间备份宣称成功。
- 最近完整外部来源参考：[0098报告](../../docs/进展报告/0098-T-26111307-原生跨来源联合评价完整收口.md)、[R72完整联合表](report/exploration-r72-reds-joint/tables.md)/[决定](docs/exploration-r72-decision.md)。四REDS源原720p/24fps、152质量与156感知行、16固定图actual review及原协调均正常结束。强自身rec仍须保留；这是合成低光DEV，不是自然夜间/独立最终，不再扩大参考家族或损失网格。
- 完整传统与已发表参考：[0096](../../docs/进展报告/0096-T-26111306-传统对照写入论文与迁移版本记录复核.md)、[R70传统参考决定](docs/exploration-r70-decision.md)，以及既有R53/R56/R65共同适配。原方法/预算/收敛限制仍适用，不把适配或负结果当新贡献。
- 论文当前为已实际审阅build018、22页/14表/12引用的内部证据稿；新增完整R72原生REDS双seed19行表及R73原生状态双seed全部注册参考24行表；后者加入时原13表不变，0编译warning/未解析引用。来源24fps合成DEV与旧30fps表分开，公开预训练像素史仍未完全核实。创新、自然动态配对、手机和独立最终未完成；[构建说明](../../paper/BUILDING.txt)现要求五生成器顺序，不录R74部分结果。

以下索引和后半部为历史阶段记录，所述“最新/正在运行/下一步”仅对应原记录日期，均服从上述接续入口和真实runtime。R44及其他旧任务已结束，旧止损决定不回写。

- R28历史状态：**前期探索，post_transport收口验证失败，已冻结主方法投入**（2026-11-12）；没有确定最终方法，转向公共动态训练资料与骨干计算分配，实拍白天ND开发集不代表自然夜景，完整独立基准/手机方法评价/时延仍缺
- 当前研究入口：[MobileIE 架构、训练核查与视频研究设计](docs/architecture-and-video-research-plan.md)
- 最新结论：[进展报告0043](../../docs/进展报告/0043-T-26111204-内部时间算子双种子闭环与路线止损.md)、[路线决策](docs/exploration-r28-decision.md)、[论文证据表](report/paper-main-table/README.md)；R28重点双seed/双目标4000步及4416行主评价完成，主门槛未通过
- 完成闭环：[原注册协议](docs/exploration-r28-protocol.md)、[全表](report/exploration-r28/tables.md)、[质量—时序图](report/exploration-r28/native_quality_stability.png)、[312行共同CPU外部表](report/exploration-r30/tables.md)。actual需要历史/flow/warp，current单帧映射单独评价；不继续沿该算子追加训练/相似网格
- 接续准备：[R32公共训练池协议](docs/exploration-r32-data-protocol.md)/[57视频3428帧登记](data/exploration-r32-training/selection.json)已排除三开发片，[完整清单](data/exploration-r32-training/manifest.json)与[解码核验](results/exploration-r32-decode/audit.json)均完成；[未训练空间预算](report/architecture-budget/tables.md)与[门控/流式源码阅读](../5.prior-art-survey/docs/25-compute-allocation-and-streaming-state.md)只约束下一方案，不是质量或手机速度结果
- 当前接续：[0051](../../docs/进展报告/0051-T-26111208-固定混合资料闭环与训练参数化边界.md)、[R40完整双域表](report/exploration-r40-mixed-anchor/tables.md)/[图](report/exploration-r40-mixed-anchor/cross_domain.png)/[决策](docs/exploration-r40-decision.md)。8000训练更新/60新质量行+96参考正常完成，固定混合呈现跨域取舍但门槛失败；停止比例/loss/LR网格。下一有限工作核查原模型训练参数化强对照，先登记函数保持/BN/fold与预算；不是新方法。
- 当前终态：[0048](../../docs/进展报告/0048-T-26111207-动态骨干双种子闭环与公开方法迁移收口.md)：R34全部24000更新、120行质量与42次CPU成本正常结束，[止损决策](docs/exploration-r34-decision.md)冻结低分辨率状态+可分离body局部扩展；[完整质量](report/exploration-r34/tables.md)/[CPU成本](report/exploration-r34-runtime/tables.md)均为开发/服务器结果，无手机测量。
- 最新强基线与来源边界：[0045](../../docs/进展报告/0045-T-26111205-自驱运动强后处理与公开预训练重叠.md)、[同底座自驱post全表](report/external-baselines-self-motion/tables.md)。双seed自驱运动post无需独立teacher，均值23.817dB/时序7.449；CPU驻留流水线200.612ms@480×854，不是手机延迟。R35三开发片与官方FastDVDnet文档训练ID重叠，不能当未见视频泛化；另30 test-dev来源只读元数据、未读JPEG。R34首次训练前已修订为120评价行，训练预算不变，stage002失败已留档；stage003接续实际训练。
- 已止损原型：[R33协议](docs/exploration-r33-prototype-protocol.md)/[功能检查](report/exploration-r33-prototype/checks.json)曾通过CPU/CUDA功能门槛；后续完整R34未过质量门槛，功能通过不等于有效方法。
- 已完成协议：[R34三控制训练](docs/exploration-r34-training-protocol.md)、[CPU成本](docs/exploration-r34-runtime-protocol.md)、[R35公开方法迁移](docs/exploration-r35-published-transfer-protocol.md)。R35完整36行及[同底座五方法表](report/external-baselines/tables.md)已归档；与R34训练底座不同，公开预训练来源重叠，不能合并为统一泛化排名。
- 外部开发资料：[R30 DAVIS协议](docs/exploration-r30-protocol.md)/[6条件缓存](results/exploration-r30-cache/index.json)，3独立视频各64帧、合成低光，VAL图像未读；[R31手机协议](docs/exploration-r31-protocol.md)/[3设备输入](results/exploration-r31-data/index.json)/[原亮度预览](report/exploration-r31-data/phone_inputs.png)，无同步GT。R30候选双seed评价已完成，手机输入尚无方法质量结论，均非完整独立基准或手机时延实测
- 运行资产：[迁移过程](process/asset-migration/journal.md)、[路径规范](../../specs/项目规范/路径与环境规范.md)；独立环境与运行数据已于2026-11-12 03:06迁入 `/data/Users/zrh/data/MobileIE-Neo`，2106文件摘要一致、16帧CPU参考指标差0，仓库链接已入Git；后续统一用 `./scripts/project-python`
- 既有完整结果：[R26监督目标控制](../../docs/进展报告/0038-T-26111200-监督目标全评价与历史贡献分离.md)、[同权重强后处理](../../docs/进展报告/0039-T-26111200-同权重单帧后处理排除聚合伪收益.md)、[R27冻结探针](../../docs/进展报告/0040-T-26111201-FST内部时间算子冻结探针与实现门槛.md)；R29潜矩资料归档附录，不继续扩大逐bit诊断
- 已知信息诊断：[进展报告0028](../../docs/进展报告/0028-T-26111121-可用时间信息与非线性聚合顺序.md)、[信息图](report/exploration-r19/known_temporal_information.png)、[顺序图](report/exploration-r19-aggregation/aggregation_order.png)、[过程](process/exploration-r19/journal.md)
- 配对反证：[进展报告0027](../../docs/进展报告/0027-T-26111121-监督配对几何反证与数据准备.md)、[图](report/exploration-r18/pair_geometry.png)、[过程](process/exploration-r18/journal.md)，104项未支持大整体整数平移主因
- 第十六轮：[进展报告0026](../../docs/进展报告/0026-T-26111121-多尺度骨干与独立历史反证.md)、[完整表](report/exploration-r16/tables.md)、[历史反证图](report/exploration-r16/multiscale_history_controls.png)、[过程](process/exploration-r16/journal.md)，24000更新/3680新逻辑行完成
- 数据准备：[R17协议](docs/exploration-r17-protocol.md)已登记，SDSD下载配额限制；BVI-RLV公开元数据可读、像素需IEEE登录，均尚无新像素结果。下一步检验时间冗余与噪声相关性
- 第十五轮：[进展报告0025](../../docs/进展报告/0025-T-26111120-公开增强权重迁移与数据域核查.md)、[公开clean增强迁移表](report/exploration-r15/tables.md)、[过程](process/exploration-r15/journal.md)
- 第十四轮：[进展报告0024](../../docs/进展报告/0024-T-26111120-原生训练与深度交互对照.md)、[深度/尺度全表](report/exploration-r14/tables.md)、[图](report/exploration-r14/native_depth_controls.png)、[过程](process/exploration-r14/journal.md)，42000新更新及全部Adam/checkpoint审核完成
- 第十三轮：[进展报告0023](../../docs/进展报告/0023-T-26111119-前置历史与训练状态分布反证.md)、[表](report/exploration-r13/tables.md)、[过程](process/exploration-r13/journal.md)
- 第十二轮：[进展报告0022](../../docs/进展报告/0022-T-26111119-已发表去噪对照与原生窗口.md)、[完整表](report/exploration-r12/tables.md)、[图](report/exploration-r12/published_denoiser_transfer.png)、[过程](process/exploration-r12/journal.md)
- 第十一轮：[进展报告0021](../../docs/进展报告/0021-T-26111119-强优化与历史对应反证.md)、[同权重历史图](report/exploration-r11/strong_history_controls.png)、[表](report/exploration-r11/tables.md)
- 第十轮：[进展报告0020](../../docs/进展报告/0020-T-26111118-原生尺度与匹配区域诊断.md)、[平滑权衡](report/exploration-r10/matched_smoothing.png)
- 第九轮：[进展报告0019](../../docs/进展报告/0019-T-26111118-优化基线与新增开发集.md)、[完整表](report/exploration-r9/tables.md)、[优化图](report/exploration-r9/optimizer_controls.png)、[过程](process/exploration-r9/journal.md)
- 第八轮：[进展报告0018](../../docs/进展报告/0018-T-26111118-输入时域化与优化强对照.md)、[完整表](report/exploration-r8/tables.md)、[优化/历史图](report/exploration-r8/optimization_history.png)、[过程](process/exploration-r8/journal.md)
- 第七轮：[进展报告0017](../../docs/进展报告/0017-T-26111118-通道状态与运动强对照.md)、[同状态与motion表](report/exploration-r7-aligned/tables.md)、[过程](process/exploration-r7/journal.md)
- 第六轮：[进展报告0016](../../docs/进展报告/0016-T-26111117-局部噪声与RGB通路辨识.md)、[条件矩与RGB表](report/exploration-r6/tables.md)、[机制图](report/exploration-r6/noise_mechanism.png)、[过程](process/exploration-r6/journal.md)
- 第五轮：[进展报告0015](../../docs/进展报告/0015-T-26111116-因果骨干与同容量反证.md)、[完整控制表](report/exploration-r5-combined/tables.md)、[历史替代图](report/exploration-r5-combined/history_mechanism.png)、[过程](process/exploration-r5/journal.md)
- 持续目标：[远期论文目标记录](docs/long-term-goal.md)
- 第四轮：[进展报告0014](../../docs/进展报告/0014-T-26111115-真实数据覆盖与时序强基线.md)、[两域结果](report/exploration-r4/tables.md)、[逐片增量](report/exploration-r4/paired_clip_deltas.json)、[完整过程](process/exploration-r4/journal.md)
- 第三轮：[进展报告0013](../../docs/进展报告/0013-T-26111102-骨干时间交互与真实视频反例.md)、[受控训练表](report/exploration-r3-training/training_tables.md)、[实拍样例](report/exploration-r3-training/real_tables.md)
- 第二轮：[进展报告0012](../../docs/进展报告/0012-T-26111021-训练对照与独立场景探索.md)、[训练/状态表](report/exploration-r2/tables.md)、[训练+融合强基线](report/exploration-r2-fusion/tables.md)
- 首轮结果：[进展报告0010](../../docs/进展报告/0010-T-26111018-首轮调研与预实验.md)、[表格与图](report/exploration-r1/tables.md)
- 上游：共享代码与权重在项目根目录 [`MobileIE/`](../../MobileIE)
- 前置工作：[2 号复现](../2.reproduction/README.md)、[3 号闪烁基线](../3.video-flicker/README.md)、[5 号前期调研](../5.prior-art-survey/README.md)
- 进展报告：[0007-T-261105-4 号工作启动与文献核查](../../docs/进展报告/0007-T-261105-4号工作启动与文献核查.md)

## 1. 本工作的判断

3 号工作保存的历史结果为：合成视频叠加噪声与压缩后，MobileIE 输出残差 **18.07/255**、输入放大 **5.5×**；旧实现的光流 EMA 为 **4.83**。2026-11-10 代码核查发现平移对齐保留回卷边界、光流方向与采样不匹配，以及有损缓存等问题；这些数字保留作历史记录，正式对照需修正后以新实验身份重跑。它们不代表真实视频的正式结论。

本工作必须回答：

1. **后处理是否已经够用？** —— 需要给出后处理在快速运动、遮挡/视差、延迟约束下的失效边界；
2. **原模型哪些结构与训练假设限制视频处理？** —— 用原论文/代码核查、受控实验与等预算消融定位。
3. **训练或结构重设计能否改善画质—稳定性—效率权衡？** —— 与普通去噪、时序损失、后处理和视频增强模型比较，以可重复收益支撑论文贡献。

2026-11-10首轮探索已保存891条方法—序列评估（54种不同输入条件，3个已有场景）。固定光照下“低维门控状态+运动补偿”有局部互补收益，但目标光照变化时固定组合劣于单用运动补偿；输入固定双边滤波也出现明显画质损失。下一轮继续调研和预实验，探索状态适应、噪声/特征变换及可靠时间冗余，不锁定唯一方法或组合。所有结果仍是受控合成预实验。

第二轮增加6张训练/6张验证素材、12次400步受控微调及光照/遮挡/切换条件。更强时序训练可接近当前有界FST收益，后者增加约35%的A100 eager前向时间，因而降低该具体实现优先级。新增普通训练+时间融合强基线，并保留特征噪声相关性/跨帧交互为待验证结构假设；数学微验证不是模型效果。共享GPU启动器会拒绝占用卡，本轮发生过一次真实资源拒绝并换空闲卡执行。

第三轮完成8次同预算微调、FST真实特征统计、门控/输出状态解耦及真实视频样例探索。body交互在原合成集配合相同融合约有0.3dB增量，但真实样例没有迁移；直接低光FB光流也出现高置信错配。当前优先深化真实训练数据、运动可靠性与曝光响应，再扩大骨干候选。增强域光流改善了时间指标但仍有细节/计算代价，不把固定融合当作已解决方案。

第四轮扩大到12训练/6测试片段，完成14次400步训练及数据覆盖、混合梯度、尺度/映射诊断。少量实拍微调的过拟合可由更多覆盖部分缓解，但均值提升不均匀，旧合成微调仍是强参考。训练选alpha0.1及固定半尺度motion将本机串行成本约82.6→34.4ms，必须纳入骨干候选对照；固定photo/census/启发式noise-scale门控未统一解决两域问题。OLiVES为白天ND衰减实拍，不等同自然夜景。

第五轮及两项必要跟进完成20次1000步训练、38条件上的3040模型评价。空间post相对plain实拍+.677dB，历史post相对同位空间post仅+.00023dB；历史替代和合成运动时序监督均未建立该分支的独立历史收益。固定TSM在切换首帧明显退化，时序训练也有跨域质量代价。降低这些具体历史实现优先级，下一步分解FST条件噪声矩、均值偏移与波动，保留强空间/廉价后处理基线。全部仍属探索。

第六轮以24静态条件、独立64校准/64评价draw，验证局部矩与RGB协方差精确分解。body二阶局部能量约68%，经门控/tail后的主要RGB代数分量却是body线性随机响应；保持均值修正减少局部方差仍未统一降低RGB波动。下一步在训练素材核验有效噪声通路及通道选择，避免按大特征能量盲目改平方算子；该诊断尚不能推广到运动或自然夜景。

## 2. 当前产出

| 产出 | 位置 | 说明 |
| --- | --- | --- |
| 第六轮条件矩与RGB通路辨识 | [R6](docs/exploration-r6-protocol.md)、[R6b](docs/exploration-r6b-protocol.md)、[R6c](docs/exploration-r6c-protocol.md)、[结果](report/exploration-r6/)、[过程](process/exploration-r6/) | 独立oracle、局部/输出噪声分离、完整门控/截断协方差及实际资源拒绝 |
| 第五轮因果骨干与同容量反证 | [R5协议](docs/exploration-r5-protocol.md)、[R5b](docs/exploration-r5b-protocol.md)、[R5c](docs/exploration-r5c-protocol.md)、[完整结果](report/exploration-r5-combined/)、[过程](process/exploration-r5/) | 20000更新、历史替代/关闭、同位空间控制、事件响应、源码/GPU审计 |
| 第四轮数据与廉价时序强控制 | [协议](docs/exploration-r4-protocol.md)、[结果/图](report/exploration-r4/)、[过程](process/exploration-r4/) | 5600更新、相同混合输入梯度控制、固定test、传统可靠性与完整处理成本 |
| 第三轮机制与真实视频反例 | [协议](docs/exploration-r3-protocol.md)、[特征/状态图](report/exploration-r3/)、[训练与真实样例](report/exploration-r3-training/)、[过程](process/exploration-r3/) | 相同初始化/信息量控制、history_off、真实低光样例、运动可靠性及状态成本 |
| 第二轮训练与广度验证 | [协议](docs/exploration-r2-protocol.md)、[训练表](report/exploration-r2/)、[融合表](report/exploration-r2-fusion/)、[过程](process/exploration-r2/) | 同初始化、同数据流训练；强正则/后处理控制；成本、反例与GPU准入审计 |
| 首轮调研与预实验 | [协议](docs/exploration-r1-protocol.md)、[图表](report/exploration-r1/)、[过程记录](process/exploration-r1/) | 有效对齐/反向warp校验、单项及组合矩阵、正负结果与特征探针 |
| 架构与视频研究设计 | [`docs/architecture-and-video-research-plan.md`](docs/architecture-and-video-research-plan.md) | FST/HDPA/重参数化、warm-up/IWO、结构候选与强基线矩阵 |
| 文献与新颖性核查 | [`docs/literature-and-novelty.md`](docs/literature-and-novelty.md) | MobileIE 引用追踪 + 相邻工作 + 选题定位 |
| 评价协议（草案） | [`docs/evaluation-protocol.md`](docs/evaluation-protocol.md) | 数据集、指标、成功判据（待导师确认） |
| 模块消融方案 | [`docs/ablation-plan.md`](docs/ablation-plan.md) | 逐模块控制变量实验矩阵 |
| 消融变体工厂 | [`scripts/model_variants.py`](scripts/model_variants.py) | 6 个变体，已通过 slim 等价性冒烟测试 |
| 方法候选与方案 A | [`docs/method-candidates.md`](docs/method-candidates.md) | 三条路线、损失设计、待定问题 |
| 时序损失原型 | [`scripts/temporal_loss.py`](scripts/temporal_loss.py) | flow warp + mask + detach，冒烟测试通过 |
| 闪烁可见性/频谱分析 | [`scripts/flicker_visibility.py`](scripts/flicker_visibility.py)、[`results/visibility/`](results/visibility) | 像素级残差、平坦/纹理区、时间标准差、时间频谱 |
| 复跑的真实退化实验 | [`results/rerun/`](results/rerun) | 用于分析的缓存帧与指标（60 帧，CPU） |

## 3. 可见性分析初步结论（2026-11-05）

针对真实退化条件（泊松噪声 + 读出噪声 + 逐帧 JPEG）：

| 指标 | 输入 | MobileIE 输出 | 放大 |
| --- | --- | --- | --- |
| 像素对齐帧差均值 | 3.23 | **18.24** | 5.6× |
| 像素对齐帧差中位数 | 2.33 | **15.33** | — |
| p95 / p99 | 9.0 / 15.0 | **42.7 / 56.3** | — |
| 超过 8/255 的像素比例 | 6.7% | **77.0%** | — |
| 超过 16/255 的像素比例 | 0.7% | **47.2%** | — |
| 逐像素时间标准差 | 2.69 | **15.14** | 5.6× |
| 平坦区 / 纹理区残差 | 3.63 / 2.90 | **18.14 / 18.56** | — |
| 时间噪声 >3 Hz 功率占比 | 80.5% | **80.4%** | — |

结论：

1. 闪烁**不是少数像素的离群值**，而是空间上普遍存在（77% 像素超过 8/255）；
2. 平坦区与纹理区同样严重——平坦区抖动在视觉上更易被察觉，说明“只处理纹理”的思路不够；
3. 时间噪声呈**宽带**（80% 功率在 3 Hz 以上、主导频率接近 Nyquist），符合“逐帧随机噪声被网络放大”，而不是慢速亮度脉动；
4. 这解释了导师“肉眼看视频不够明显”的观感：残差虽大，但空间上随机、时域上高频，需要帧差放大/局部裁剪/时序曲线才能稳定展示。

图表：`results/visibility/{series,hist,temporal_std_maps,spectrum}.png`。

## 4. 工作计划

以下保留2026-11-05的阶段计划；当前属于前期探索，实际优先级及下一轮假设以最新进展报告0016为准，暂不把方法候选当成已确定的正式实验路线。

### P0（本周，服务器到位前）
- [x] MobileIE 引用追踪与相邻工作检索；
- [x] 评价协议草案、模块消融方案；
- [x] 基于现有数据的可见性/频谱分析；
- [ ] 与导师确认“解决”的量化判据；
- [ ] 真实视频素材收集（自拍 + 公开数据集，待用户提供/确认）。

### P1（11/7 服务器到位后 2–3 周）
- [ ] 迁移到 8×A100 服务器，完成正式训练 baseline；
- [ ] 后处理边界实验：速度扫描、遮挡/视差、延迟；
- [ ] 执行逐模块消融（见 `docs/ablation-plan.md`）。

### P2（12 月，方法原型）
- [x] 方案 A 损失原型（推理零开销，冒烟测试通过）；
- [ ] 方案 A 正式训练与 λ 调参；
- [ ] 方案 B：轻量帧间模块（feature EMA / 1–2 帧输入）；
- [ ] 方案 C：输入多帧拼接（对照）；
- [ ] 统一比较闪烁、画质、参数量、MACs、延迟、快速运动鲁棒性。

### P3（1–2 月，收尾）
- [ ] 手机端 TFLite / Android 模拟器验证；
- [ ] 论文实验与写作（目标 ICCV 2027，3 月 8 日截止）。

## 5. 复现

```bash
PY=F:/dev-tools/python-envs/MobileIE-Neo/Scripts/python.exe

# 重跑真实退化实验（生成缓存帧 + 指标）
"$PY" research/3.video-flicker/scripts/flicker_realistic.py --device cpu \
  --out "F:/资源夹/科研/SQS Team/MobileIE-Neo/research/4.temporal-consistency/results/rerun"

# 可见性 / 频谱分析（依赖上面的缓存帧）
"$PY" research/4.temporal-consistency/scripts/flicker_visibility.py
```

> 缓存帧写在 `F:/pip-tmp/mobileie_frames_real`（ASCII 路径，规避中文路径问题）。

## 6. 修订记录

- 2026-11-05 · claude code (local)：创建 4 号工作，完成文献核查、评价协议、消融方案与可见性分析。
- 2026-11-10 18:03 +08:00 · Codex  (sqs_A100)：以论文目标更新主线入口，标注历史指标待重核，并将结构设计、训练机制和完整基线纳入研究问题。
- 2026-11-10 18:29 +08:00 · Codex  (sqs_A100)：记录首轮自主探索已完成，连接逐序列结果、失败资料和进展报告，保持多路线及组合开放。
- 2026-11-10 21:13 +08:00 · Codex  (sqs_A100)：连接第二轮受控训练、独立素材验证、强基线与资源审计；据成本和对照结果调整候选优先级。

修改：2026-11-11T16:40:43.250064+08:00；作者：Codex  (sqs_A100)；原因：更新0015结果与持续研究入口，历史阶段计划保留供追溯。

修改：2026-11-11T17:09:03.253353+08:00；作者：Codex  (sqs_A100)；原因：索引0016及已完成噪声诊断。

修改：2026-11-11T18:25:49.244614+08:00；作者：Codex  (sqs_A100)；原因：索引0017/0018及优化强对照研究。

修改：2026-11-11T18:37:56.584049+08:00；作者：Codex  (sqs_A100)；原因：索引0019强优化基线，限定低LR结构结论范围。

修改：2026-11-11T18:51:23.382324+08:00；作者：Codex  (sqs_A100)；原因：索引0020原生诊断，记录R11强优化历史评价进行中。

修改：2026-11-11T19:15:58.253077+08:00；作者：Codex  (sqs_A100)；原因：索引0021/0022，记录R13前置历史评价中。

修改：2026-11-11T19:42:53.736037+08:00；作者：Codex  (sqs_A100)；原因：索引0023及R14深度训练启动，保留完整研究目标。

修改：2026-11-11T20:24:45.731053+08:00；作者：Codex  (sqs_A100)；原因：索引R14/R15完整结果及新的结构/数据问题。

修改：2026-11-11T20:48:02.482890+08:00；作者：Codex  (sqs_A100)；原因：索引R16结构训练与R17数据获取限制。

修改：2026-11-11T21:13:08.692013+08:00；作者：Codex  (sqs_A100)；原因：索引0026与固定配对诊断，记录阶段推送已恢复。

修改：2026-11-11T21:21:00.783613+08:00；作者：Codex  (sqs_A100)；原因：索引0027几何负结果与后续时间冗余问题。

修改：2026-11-11T21:48:32.613420+08:00；作者：Codex  (sqs_A100)；原因：索引0028及动态聚合强对照，保持论文目标进行中。

修改：2026-11-11T22:09:32.987886+08:00；作者：Codex  (sqs_A100)；原因：索引0029动态强控制与R21聚合感知训练。

- [0030 聚合训练完成、评价等待](../../docs/进展报告/0030-T-26111122-聚合感知训练完成与共享资源等待.md)：R21完整32000更新/32检查点审计通过；共享GPU被占用，5888行评价待空闲，不作质量声明。

修改：2026-11-11T22:29:57.739588+08:00；作者：Codex  (sqs_A100)；原因：索引R21真实完成状态、资源等待与后续机制诊断。

[R22/报告0031](../../docs/进展报告/0031-T-26111122-FST非线性与插值交换误差诊断.md)：24固定CPU算子诊断通过，非质量/新方法结论；R21全部候选与强参考已切换共同CPU评价，进行中。

修改：2026-11-11T22:35:57.291566+08:00；作者：Codex  (sqs_A100)；原因：索引已完成R22与R21评价后备状态。

[0032 聚合训练后静态强对照](../../docs/进展报告/0032-T-26111122-聚合训练后的静态信息利用强对照.md)：384同CPU统计/80精确重放完成；历史利用与优于经典基线分开，动态R21仍评价中。

修改：2026-11-11T22:50:29.287295+08:00；作者：Codex  (sqs_A100)；原因：索引固定静态终点诊断与负结果。

[0033 聚合训练动态全评价](../../docs/进展报告/0033-T-26111122-聚合训练动态全评价与强后处理边界.md)：5888候选+2208同CPU参考、880源码SHA、44短片序列及逐帧回放完成；无跨域一致强基线优势。下一步分解全局RGB偏差/空间误差，论文目标active。

修改：2026-11-11T22:59:08.983826+08:00；作者：Codex  (sqs_A100)；原因：索引R21完整结果与后续聚焦方向。

[0034 全局RGB与空间误差](../../docs/进展报告/0034-T-26111123-全局RGB偏差与空间误差分解.md)：44固定序列/704帧精确分解完成，正式PSNR全帧口径已澄清；下一步R24冻结global att干预，零新训练，不称新方法。

修改：2026-11-11T23:10:29.544887+08:00；作者：Codex  (sqs_A100)；原因：索引精确误差诊断与结构节点检验。

[0035 冻结全局通道读出](../../docs/进展报告/0035-T-26111123-冻结全局通道读出干预.md)：96配置/640 body等值/64正式精确回放完成；单节点不能统一补回主质量差，接续R25同接受域对应敏感性。

修改：2026-11-11T23:19:21.655498+08:00；作者：Codex  (sqs_A100)；原因：索引有界结构干预及下一诊断。

[0036 同接受域对应与目标转向](../../docs/进展报告/0036-T-26111123-同接受域对应选择与训练目标转向.md)：160配置/80正式精确回放完成，GT估计flow未恢复主要PSNR差；R26已启动48k同骨干Charb/MSE强控制，最终素材保留。

修改：2026-11-11T23:42:37.657434+08:00；作者：Codex  (sqs_A100)；原因：索引对应敏感性负结果与深度训练目标对照。

[0037 监督目标训练完成](../../docs/进展报告/0037-T-26111200-监督目标控制训练完成与参考精确重放.md)：48k更新/48检查点完成；Charb全部24模型/Adam检查点精确重放，8832行共同CUDA评价中，尚无质量结论。

修改：2026-11-12T00:03:31.017038+08:00；作者：Codex  (sqs_A100)；原因：索引已审计训练与待完成评价，区分两阶段。

[0038 监督目标全评价](../../docs/进展报告/0038-T-26111200-监督目标全评价与历史贡献分离.md)：8832行及1104序列/23808帧历史精确重放完成。原生pre/MSE增益主要也出现在current映射；同骨干后处理强控制R26b已运行，待审计。

修改：2026-11-12T00:48:20.998508+08:00；作者：Codex  (sqs_A100)；原因：索引完整监督目标评价及严格历史贡献边界。

[0039 同权重强后处理](../../docs/进展报告/0039-T-26111200-同权重单帧后处理排除聚合伪收益.md)：5888行/2944序列与63488帧精确回放；原生MSE两聚合均被自身单帧+post四项域均值指标支配，接续FST内部算子控制。

修改：2026-11-12T00:51:44.042967+08:00；作者：Codex  (sqs_A100)；原因：索引R26b完整证据与结构探索转向。

[0040 FST内部算子](../../docs/进展报告/0040-T-26111201-FST内部时间算子冻结探针与实现门槛.md)：240冻结行/200前向、current与8段128帧CPU参考精确重放；五4075参数算子已实现，CUDA门槛等待共享卡空闲，尚未训练。首copy Adam失败未定位并完整保留。

修改：2026-11-12T01:10:55.655899+08:00；作者：Codex  (sqs_A100)；原因：索引原算子深度探索及当前边界。

修改：2026-11-12T05:10:57+08:00；作者：Codex  (sqs_A100)；原因：索引受预算原型、已登记训练门槛及正在执行的同底座公开方法评价。

修改：2026-11-12T05:39:58+08:00；作者：Codex  (sqs_A100)；原因：索引强post实测、公开训练重叠和训练前修订。

修改：2026-11-12T06:08:00+08:00；作者：Codex  (sqs_A100)；原因：索引真实数据/执行门槛和训练前batch口径修正。

修改：2026-11-12T06:09:14+08:00；作者：Codex  (sqs_A100)；原因：替换入口顶部已过时的下载/CUDA等待状态。

修改：2026-11-12T06:35:24+08:00；作者：Codex  (sqs_A100)；原因：索引首seed闭环及固定成本测量准备，不把活动输出当最终。

修改：2026-11-12T07:17:16+08:00；作者：Codex  (sqs_A100)；原因：以完整结果更新接续入口，保留历史报告。

修改：2026-11-12T07:34:27+08:00；作者：Codex  (sqs_A100)；原因：索引传统参考提升与已实际启动的新机制检验。

修改：2026-11-12T07:57:36+08:00；作者：Codex  (sqs_A100)；原因：使用完成终态接续，新增真实输入与有GT域转移约束。

修改：2026-11-12T08:23:58+08:00；作者：Codex  (sqs_A100)；原因：索引已结束的固定资料检验与原训练范式边界。

修改：2026-11-12T13:05:50.485664+08:00；作者：Codex  (sqs_A100)；原因：索引R41实际训练与失败边界，保留已完成资产迁移。

修改：2026-11-12T13:30:13.952419+08:00；作者：Codex  (sqs_A100)；原因：索引R41完整收益及失败边界、R42固定已知空间对照。

修改：2026-11-12T13:48:57.243894+08:00；作者：Codex  (sqs_A100)；原因：索引R42完整止损与R43有界原型，不重启已结束作业。

- 最新：[0055原型/标定](../../docs/进展报告/0055-T-26111214-FST根部原型与训练尺度标定.md)、[R44固定训练协议](docs/exploration-r44-bounded-video-training-protocol.md)；无新正式训练/质量。

修改：2026-11-12T14:06:10.126056+08:00；作者：Codex  (sqs_A100)；原因：区分R43实际完成与R44登记，避免重复或提前启动GPU作业。

- 正在运行：[0056启动报告](../../docs/进展报告/0056-T-26111214-根部激活与时间目标四组训练实际启动.md)、[R44过程](process/exploration-r44/journal.md)。

修改：2026-11-12T14:26:45.433677+08:00；作者：Codex  (sqs_A100)；原因：将R44登记状态更新为已验证的实际训练，保持旧阶段冻结。
