# MobileIE-Neo

面向手机端的轻量图像增强模型研究：在 MobileIE（极轻量实时图像增强 ConvNet）基础上，解决其应用于视频时产生的帧间闪烁问题。

## 当前研究状态

更新：2026-11-21T15:45:42.346792+08:00；Codex  (sqs_A100)。最新[0207：八帧稳定化基线资格与双种子学习](docs/进展报告/0207-T-26112115-八帧稳定化基线资格与双种子正式学习.md)已通过CPU/CUDA资格，两个正式种子在GPU6/7运行；固定两个已发表标准层级、各80K更新，四唯一final后完整评价。真实连续8帧采样已落实，无新DEV画质或论文方法结论；资格权重不继承，按实际原句柄接续并保留全部旧止损。论文目标active未完成。

更新：2026-11-21T15:09:20.799735+08:00；Codex  (sqs_A100)。最新[0206：已发表稳定化基线接入与分辨率费用](docs/进展报告/0206-T-261121-已发表稳定化基线接入与分辨率费用评估.md)补入NeurIPS2025 IVM原代码，有限CPU资格通过，但完整层控制器在原生MobileIE上的解析卷积和状态代价高。未启动正式训练或声称质量收益；下一登记同论文简单学习衰减基线，保留完整成本与既有止损。当前无活跃I1/GPU，论文目标active未完成。

更新：2026-11-21T14:49:32.684696+08:00；Codex  (sqs_A100)。最新[0205：固定未来创新平滑完整结果](docs/进展报告/0205-T-261121-固定未来创新平滑完整结果与止损.md)在research/O1完成120序列/5016逐帧，必要门6/16、完整失败。SYN对原因果强基线有正例，但相对同未来简单混合全部20条件时间更差；[固定方案停止](research/8.video-backbone/fixed-lag/decision.md)，不追加救分训练或成本/手机评价。全部图审/正例归档、原任务终态，0新CNN/训练/GPU；一未来与经典平滑不当零延迟或创新，论文目标active未完成。

更新：2026-11-21T14:23:17.431948+08:00；Codex  (sqs_A100)。最新[0204：来源时序核查](docs/进展报告/0204-T-261121-来源帧索引与时间基准核查.md)完成76来源组/32窗口及既有手机解码时间记录核查；本地未作时间跳帧，来源名义帧率与实际时钟分级声明，不能把所有源统一当实际30fps。无新像素/训练/GPU，旧指标与失败配置不改；下一回到递归平滑的科学前提，论文目标active未完成。

更新：2026-11-21T14:08:42.423153+08:00；Codex  (sqs_A100)。最近实验[0203：一帧等待信息筛查](docs/进展报告/0203-T-261121-单帧等待信息筛查完整结果与止损.md)完成120序列/5016逐帧、0新CNN/训练/GPU。未来观察对同数量过去观察有质量/时间正增量，但完整必要门3/16失败，[固定配置停止](research/8.video-backbone/delay-budget/decision.md)，全部图审/局部正例保留。下一先核采集/提取时序，未来等待不当零延迟；原任务终态，论文目标active未完成。

更新：2026-11-21T13:45:14.512113+08:00；Codex  (sqs_A100)。此前[0202：同输入证据归并](docs/进展报告/0202-T-261121-同输入证据归并与时间信息预算问题.md)完成896配对/56统计及全场景权重敏感性；历史收益不足1%对逐一留出仍成立，部分平均画质有场景权重限制。本轮无新训练/像素/GPU，原失败配置保持STOP；下一为明确一帧等待的信息预算问题，尚未登记科学协议或运行。论文目标active未完成。

更新：2026-11-21T13:29:13.971968+08:00；Codex  (sqs_A100)。最近实验[0201：局部仿射算子缓存筛查与止损](docs/进展报告/0201-T-261121-局部仿射算子缓存筛查与止损.md)：research/O1上的[A1](research/8.video-backbone/affine-cache/README.md)全部64缓存窗/768输出/48均值完整，0新CNN/训练/GPU，必要门0/20；每帧投影已损失SYN质量，隔帧复用进一步退化。[固定配置停止](research/8.video-backbone/affine-cache/decision.md)，局部正例/全部图审保留，不扩失败训练/成本/DEV/手机。原任务终态，论文目标active未完成。

更新：2026-11-21T12:58:24.776394+08:00；Codex  (sqs_A100)。此前[0200：潜在状态完整学习评价与止损](docs/进展报告/0200-T-261121-潜在状态完整学习评价与止损.md)：research/O1上的[L1](research/8.video-backbone/latent-core/README.md)全32K正式更新/512K学生、四唯一最终及同32窗/704输出/44均值与实际全图审阅完成；空间2/8、历史0/12、强post0/24，完整A/B失败，[固定配置STOP](research/8.video-backbone/latent-core/decision.md)。平均空间提升及局部正例完整保留，不当独立时间收益。原全部作业终态，不重启；26/44＋1源、O1/所有旧STOP/主论文/封存最终/并行worktree保持，论文目标active未完成。

此前[0191：外观索引特征历史完整筛查与止损](docs/进展报告/0191-T-26112101-外观索引特征历史完整筛查与止损.md)：冻结原模型的B1完成16训练窗口/80输出/320逐帧、0优化/GPU。历史对当前空间控制仅.167%–.744%时间改善，完整门失败，[固定配置停止](research/8.video-backbone/appearance-state/decision.md)；所有局部收益/完整结果和图已保留，原进程正常终态。

此前[0190：多线性骨干完整空间恢复与止损](docs/进展报告/0190-T-26112101-多线性骨干完整空间恢复与止损.md)：M1全32000正式更新、四唯一最终及固定评价1536NN/384输出/12均值完成，八项恢复保护0/8；对F0四组PSNR低1.45–3.65dB，局部正例与其他指标取舍完整保留。[完整决定](research/8.video-backbone/multiaffine-core/spatial-decision.md)停止固定M1局部网格，不进入视频/DEV/手机；原全部句柄终态，论文目标仍active。

此前[0188：新骨干资格与空间恢复启动](docs/进展报告/0188-T-26112023-多线性骨干实现资格与空间恢复正式启动.md)和[0189：首种子与接续](docs/进展报告/0189-T-26112100-多线性骨干首种子完整训练与第二种子接续.md)保留历史阶段；当前结果以0190为准。40具名源/输入与原空间协议不改，raw多线性资格不等于clip后RGB保证；勿从旧运行提示重启。

此前[0187：三观测方差校准与骨干问题](docs/进展报告/0187-T-26112023-三观测方差识别校准与骨干问题收敛.md)：[N1](research/8.video-backbone/triple-variance/README.md)全11264NN/48组正常完成，固定photon8/8正例保留、跨photon4/16，完整实际运动资格False。增强后均值变化/相关误差不能忽略，固定估计网格STOP；无新视频质量或方法成功。两全点PNG已实际查看，[完整决定](research/8.video-backbone/triple-variance/decision.md)、[真实终态](research/8.video-backbone/triple-variance/process/terminal001.json)与[待核骨干问题](research/8.video-backbone/triple-variance/next-core-question.md)分别留存，无活跃作业。

此前[0186：单向传输筛查与止损](docs/进展报告/0186-T-26112022-单向传输贡献者离散度筛查与止损.md)：[S1](research/8.video-backbone/single-direction/README.md)已完成64NN/80输出条件、0更新/GPU/DEV/最终。三个完整门均False，对原双向DIS三个组质量低.51–.73dB，ND0和逐窗正例保留；固定目标贡献者矩/mass界/阈值/alpha/preset/field来源网格STOP。全20均值与16窗口联系图实际查看，[决定](research/8.video-backbone/single-direction/decision.md)和[真实终态](research/8.video-backbone/single-direction/process/terminal001.json)保持完整，无活跃实验，不按旧state重启。

此前[0185：粗能量历史零更新筛查与止损](docs/进展报告/0185-T-26112022-粗能量历史零更新筛查与止损.md)：[P1](research/8.video-backbone/phase-envelope/README.md)已完成1280NN/0优化/200输出条件，冻结原模型的当前载波+能量历史在四噪声组对raw时间升.456%–.560%，未通过训练门槛。clean正例保留，停止固定能量/差分/alpha/尺度/clip/epsilon网格；全30均值图和固定联系图实际查看，无新DEV/GPU/最终。[完整决定](research/8.video-backbone/phase-envelope/decision.md)、[真实终态](research/8.video-backbone/phase-envelope/process/terminal001.json)，无活跃实验，不按旧state重启。

此前[0184：部署递归监督完整评价与止损](docs/进展报告/0184-T-26112021-部署递归监督完整评价与止损.md)：[T2](research/8.video-backbone/recurrent-objective/README.md)已完成16000正式更新/四唯一最终、120质量与120官方JOD/360共同条件/72均值、四固定图和两全点PNG实际审阅。恢复0/4、A成对0/44、B10/44、额外学习0/8，完整资格False。对独立clip-GT DIS有SYN PSNR+.10–.11dB/14条件PSNR正例，但全部20条件时间更高（四组+1.50%–2.46%），固定配置局部网格STOP，不扩失败成本/外部/手机，不继续向T1/T2加loss或延长诊断。原协调/数值/评分任务全部终态、GPU释放，勿按旧静态PID重启。[完整决定](research/8.video-backbone/recurrent-objective/decision.md)、[主表](research/8.video-backbone/recurrent-objective/report/joint/tables.md)、[真实终态](research/8.video-backbone/recurrent-objective/process/terminal001.json)。

此前[0180：同原骨干时序KD完整评价与止损](docs/进展报告/0180-T-26112019-同原骨干时序KD完整评价与止损.md)：主题8-T1全部24000正式更新、六唯一最终、180质量/180官方JOD、240共同条件/48均值及四固定图实际审阅均完成。恢复保护3/4、主要A/B各0/40、成本扩展False；相对空间KD保留20/20时间改善、四组1.49%–2.38%的小幅正例，但未追平强post，固定同F0/T4DIS/等权KD局部网格STOP。普通KD已有直接先例，只作已完成已知训练参照。[完整决定](research/8.video-backbone/teacher-targets/decision.md)、[完整表](research/8.video-backbone/teacher-targets/report/joint/tables.md)、[真实终态](research/8.video-backbone/teacher-targets/process/terminal001.json)。

前序[0175：V1主干完整评价](docs/进展报告/0175-T-26112017-完整视频骨干候选评价与止损.md)已完成28000正式训练与220质量/JOD、280共表，空间保护0/4、A/B各0/28，固定packed核/输入递归局部配置已STOP。全部V1/T1/T2原数值/协调任务已终态、GPU释放；下面历史中的“下一”“当前”不作为重启依据。所有旧STOP、原84值/snapshot003/内部PDF020、独立最终封存、7号与其他worktree保持。

论文goal仍active，尚无确立创新、手机或独立最终结论。手机轻量实时需要完整成本证据，原4075参数不是用户硬性参数上限，也没有用户要求所有候选必须零额外推理；每个已冻协议仍按自身预算/门槛判定。下一独立问题须说明相对失败路线的结构/信息入口差异，先核直接先例再登记少量可证伪原型，不延长本KD救分。接续读[服务器约定](AGENTS.md)、[主题8](research/8.video-backbone/README.md)、[论文证据入口](paper/RESEARCH_STATUS.md)。代码/结果/过程/报告分别密集提交和阶段SSH推送，长等待按实测速率sleep并核原句柄。

## 历史阶段记录（不作为当前活跃状态）

此前[R87 / 0167](docs/进展报告/0167-T-26111714-平方表征保持函数校准与匹配前提止损.md)：平方表征度量校准已完整CPU执行，256骨干图/1024度量更新、0骨干更新，RGB映射数值保持而四组检索前提0/4；停止本缩放/投影网格，不当新方法或视频质量结果。下一[待核问题](research/4.temporal-consistency/docs/next-core-question-after-r87.md)转向更早共享stem中的空间信息，尚未实现/训练，需先核直接先例和完整因果设计。无活跃R87/GPU，R86两传统trade-off/论文84原质量值/paper020和旧STOP保持。

此前[R86 / 0166](docs/进展报告/0166-T-26111711-固定快速DIS经典参考完整质量与配对成本.md)：固定DIS传统参考的20质量/20 JOD、3200真实骨干配对成本与20图实际审阅均正常完成，无活跃任务。域平均PSNR/JOD有收益、同轮CPU快约16%–26%，但bear低ph单源JOD下降、ND时间误差高7%–8%，四组联合0/4；保留两经典trade-off，停止preset/尺度/阈值网格，不当新方法。[论文证据入口](paper/RESEARCH_STATUS.md)已补DIS及来源，原质量值保留，仍缺原创方法/自然动态/手机/独立最终，paper020与旧STOP保持。

此前[R85 / 0165](docs/进展报告/0165-T-26111711-现有强基线完整CPU流水线成本与快速传统参考缺口.md)：两seed/十条件原F0与strong-rec完整CPU成本已正常测完，3200真实骨干、120正式序列/40等价资格最大差0，原PID退出，0新SGD/GPU/质量指标。当前经典half-FB参考约为raw的1.79–1.91倍完整计算时间，仅本机常驻CPU前16帧，不外推手机或所有传统flow。[论文证据入口](paper/RESEARCH_STATUS.md)已补成本，原质量值保持；下一单独登记固定快速DIS经典参考，尚未有其图像calc/速度/质量结果，不改旧STOP或声称新方法，PDF020/保留最终保持。

此前[R84 / 0164](docs/进展报告/0164-T-26111711-共享特征非局部原型完整对照与止损.md)：共享原平方前特征的因果匹配原型及energy/RGB控制完整CPU运行，132骨干图/288候选对/160质量行、0SGD/DEV/GPU；接纳仅3%–6%，四组匹配/输出门槛均未过，停止本固定hard匹配局部网格，不扩大为共享可学习特征无效。已建立[论文证据入口](paper/RESEARCH_STATUS.md)，统一共同DEV双seed结果、公开参考与未测成本/手机缺口；训练窗不混主表。无活跃新训练或已确立新方法，全部旧STOP/最终像素/paper020保持。

此前[R83 / 0163](docs/进展报告/0163-T-26111710-状态算子范围检验与监督来源勘误.md)：一次四训练窗/两F0的CPU前置检验正常结束，128骨干图块/144 half-FB对、0SGD/DEV/GPU。强rec状态响应约63.5%–68.5%的系数能量在局部5×5之外，GT/F0条件下的共同域状态算子也有实质差别；两必要前提过，但不是方法/创新/视频性能结论。下一共享原head有符号特征的不同核心原型尚待实现与冻结；R82及旧STOP/保留最终/paper020保持，论文目标active。

此前R82 CPU资格[R82 / 0159](docs/进展报告/0159-T-26111706-核预测先例收口与自参照修正原型资格.md)：KPN/BPN/F2R直接先例已限定；304参数共同条件修正头通过真实训练窗CPU资格（12丢弃更新、0正式/DEV/GPU），self/流式差均0、过去梯度非零但很小，非历史有效性证据。双seed三控制共48000正式更新及完整共同评价已预登记；完整实现与CUDA资格尚待，无正式作业。旧STOP/最终像素/paper020保持。

截至2026-11-17，项目仍处于方法探索阶段，尚未建立可投稿的新方法。接续先读[进展报告0167](docs/进展报告/0167-T-26111714-平方表征保持函数校准与匹配前提止损.md)、[4号研究入口](research/4.temporal-consistency/README.md)和[服务器约定](AGENTS.md)。主线在 `main` 推进，移动演示由独立支线处理。



最新R81（[0158](docs/进展报告/0158-T-26111704-训练噪声门控干预与局部路线止损.md)/[停止决定](research/4.temporal-consistency/docs/exploration-r81-decision.md)）：一次固定训练图块静态噪声干预已正常结束，8主条件0/8。128唯一噪声图由双seed共用、272骨干/1040tail、0SGD/DEV/GPU；8固定图均已实际查看。无噪参照localgate有部分PSNR/7.5%–9.2%波动收益但不稳，global替换可能与当前特征失配，均非部署/最佳门控上界。不继续此小门控/温度/滞回/EMA等网格；R80及全部旧STOP/最终像素/paper020保持，尚无新方法。

此前R80已完整收口（[0157](docs/进展报告/0157-T-26111703-有限前缀条件读出完整负结果与先例边界.md)/[停止决定](research/4.temporal-consistency/docs/exploration-r80-decision.md)）：160新质量及160官方JOD、各6368比较帧，联合200条件/40均值/24比较通过4，tail0/12，20固定图已逐张实际查看；原CPU/指标/协调全部正常退出、31源保持，无活跃R80作业。tail相对raw时序仅降.81%–1.88%，对强rec仍高38.13%–50.96%，停止本13basis/ridge/暖机/teacher/seed局部网格。GT诊断ND PSNR虽明显改善但时间几乎不降，非部署结果；note65核少量通道适配/线性展开已有先例。尚无有效新方法，旧STOP/最终像素保护及paper020保持。

当前[R79完整决定](research/4.temporal-consistency/docs/exploration-r79-decision.md)：24000更新、100新质量/4480帧与100 JOD均正常结束，联合140条件/28域均值/16比较全部False，全部20固定图已实际查看。课程相对同权重current时间代理仅降.48%–.83%，相对强rec仍高32.68%–51.74%；正负条件/seed都保留，按协议停止本课程局部网格，不推广为所有课程无效。原CPU/感知/协调均退出，禁止据旧快照重启；21/11源未变。[note63](research/5.prior-art-survey/docs/63-fip-diagnostics-and-curriculum-novelty-boundary.md)核明FIP训练/诊断已有先例，非创新。完整结果暂独立、paper020未变；有效新方法、自然动态/手机/独立最终仍缺，保留像素/旧STOP保持。

前序定向核查[训练信息课程先例](research/5.prior-art-survey/docs/62-training-information-curricula-and-photometric-transfer.md)：FIP遮中心预训练/LVD不平衡噪声已有作者代码，不能直接作为创新。R78仅人工光度资格（8行、92/92），显示把暗输入−亮GT当噪声缩放会在无噪时也改变曝光；真实暗clean/噪声与课程有效性未被识别。该人工资格已关闭、不再追加同类网格；实际网络学习另登记为上述R79。

完整R74负结果与R75/R76最终证据已接入[论文工作稿](paper/manuscript.pdf)：build020为24页、16张数值表、13项引用，无编译warning/未解析引用，整份缩略和变化页已实际查看。旧14张数值表不变；两张新表保留全部双seed读出与八主门槛/四支持。R75八项0/8、R76支持1/4、R77停止额外投资保持；这仍是内部证据稿，创新方法、自然动态配对、手机和独立最终未完成。下一回到不同信息/学习问题的假设选择，不重开旧网格。

[原保留实拍来源](research/4.temporal-consistency/data/external-olives-reserved/README.md)只做纯目录元数据资格，三段586对帧名、0像素/模型/解码；不增加已完成独立基准数，不开放为DEV，也不因原门槛通过自动解封。最终方法、权重、预处理和完整评价范围须另冻结；原训练与既定完整评价链继续，paper018和研究缺口保持。

最近[progress032](research/4.temporal-consistency/process/exploration-r75/progress032.json)已确认原两seed各64000、共128000更新及R75/R76完整评价均正常结束，旧训练/协调进程已退出。观察间隔38K–64K检查点逐份归档，完整流、数值/感知、两套各20固定图及实际图审均已保存；17/27/37源保持。R75八联合门槛0/8，R76有限支持1/4，按预登记停止R77额外128K训练投资，无新正式任务启动。

完整[因果参考表](research/4.temporal-consistency/report/exploration-r75-joint/tables.md)为80条件/16均值，[self4补充表](research/4.temporal-consistency/report/exploration-r76-joint/tables.md)为100条件/20均值，原80条件与八门槛原样保留。ND空间PSNR较好但未建立跨条件时间优势，见[R75决定](research/4.temporal-consistency/docs/exploration-r75-decision.md)和[R76决定](research/4.temporal-consistency/docs/exploration-r76-decision.md)。current/self4是同权重输入控制，非独立训练空间模型；四次self4非同计算。

这只是发布ReBotNet-XS的自启动/原生分块任务适配，MobileIE8K/128与XS64K/384训练条件不等，不是原论文500K复现、原创方法或SOTA。SYN480×854和ND288×512原生裁块均重复DEV，非整幅1080p/自然夜间/手机/独立最终。完整证据现已接入上述build020；旧STOP与保留像素保护保持。以下启动/资格叙述仅为历史，不据此重开已结束任务。

定向来源核查已确认ReBotNet作者发布架构，CPU人工输入资格通过，原代码/许可已固定入库。固定token/GT首帧/500K原训练与任意原生分辨率的差异见[57号阅读](research/5.prior-art-survey/docs/57-causal-reference-code-feasibility.md)；代码资格阶段没有训练/质量/视频像素/GPU结果，不称方法或原论文充分复现。

前阶段另完成原生分块CPU检查和两次丢弃共享GPU更新，原试跑进程已退出释放、状态不继承；其[正式预算](research/4.temporal-consistency/docs/exploration-r75-causal-reference-protocol.md)现已由上述正式作业执行。资源资格、活跃训练与最终有质量收益的方法分别记录。

R74冻结空间映射/RGB传输的[固定检验](research/4.temporal-consistency/docs/exploration-r74-motion-supervision-protocol.md)已完整结束：16000 motion更新、80新CPU质量、80感知、20固定图实际审阅及120条件联合表全部保留。16门槛全部失败，按[完整决定](research/4.temporal-consistency/docs/exploration-r74-decision.md)停止其局部搜索，未运行成本实验；当前无活跃R74任务，不依据历史提示重启。共享GPU实际设备问题已透明纠正，后续[入口](research/4.temporal-consistency/docs/prebound-shared-gpu-entry.md)在整个子进程前绑定并核实际UUID，非0卡零评分检查通过；旧错误记录和所有原分数保持。创新和适当因果视频参考仍需继续建立，不把普通模块组合或负结果称方法成功。

R73原生一/二阶历史矩进入第一次FST的固定检验已全部正常结束：24000正式更新、六最终导出、200新CPU质量、240感知、280联合条件和20固定图实际检查完整保留。20项联合门槛全部失败，按[完整决定](research/4.temporal-consistency/docs/exploration-r73-decision.md)停止该组合的局部网格；没有因此建立新方法。当前无活跃R73任务，不依据历史报告中的活跃提示重启旧训练或评价。

R28/R34/R37/R44/R45/R48/R64等既有具体候选已按各自完整结果止损，不重开相似超参数搜索。R70传统视频滤波、R72公开REDS开发源强参考均已正常收口；这些开发证据不等于独立最终测试或自然夜间验证。CRVD仅一个训练时刻四份原生RAW文件的有限输入资格已完成，未进行模型评价，不计作完整视频基准。

现有[论文工作稿](paper/manuscript.pdf)为已审阅build018、22页/14表，包含完整R72原生REDS合成开发对照及R73原生状态主候选/全部注册参考的完整失败证据；尚缺可辩护创新、画质—时序—效率联合改善、自然动态配对、实际手机成本及独立最终验证。轻量移动端目标需要完整成本证据；原MobileIE的4075参数不是用户给定的硬性参数上限，含历史状态的模型也不能称零推理开销。

2026-11-12 03:06，自有环境和数据迁移已核验完成，两个仓库相对链接已入Git，后续统一用 `./scripts/project-python`；不重复迁移。详情见[迁移日志](research/4.temporal-consistency/process/asset-migration/journal.md)。代码、过程、结果、报告分别小步提交，阶段SSH推送，保持 `--no-ff`。新GPU任务须先检查共享卡的显存、利用率及计算进程，确认双方资源余量后可同卡并行，使用同组最多4张卡。

## 目录结构

| 目录 | 用途 |
| --- | --- |
| `MobileIE/` | 上游官方代码（vendored，提交 `7030d12`）与预训练权重，各工作共用 |
| `docs/` | 进展报告（Markdown） |
| `specs/` | 项目规范与需求文档，如 `项目规范/版本管理规范.md` |
| `research/` | 各阶段工作的中间资料与辅助内容（前期准备、复现、视频闪烁） |
| `assets/` | 图片等静态资源 |

## 项目规范

版本管理遵循 [`specs/项目规范/版本管理规范.md`](specs/项目规范/版本管理规范.md)：频繁提交、SSH 提交、AI 提交作者标注 `claude code (local)` / `Codex (local)`、禁用 fast-forward 合并。本服务器主线按用户约定使用作者 `Codex  (sqs_A100)`，覆盖旧本地署名默认。

## 快速开始

```bash
git clone git@github.com:spinner-bot/MobileIE-Neo.git
cd MobileIE-Neo
```

服务器运行资产固定在 `/data/Users/zrh/data/MobileIE-Neo`；仓库 `.venv`、`.cache` 已分别指向该目录的独立环境和缓存，并将相对链接入Git。运行统一用 `./scripts/project-python`，例如 `./scripts/project-python -c 'import torch; print(torch.__version__)'`。迁移过程及恢复记录见[迁移日志](research/4.temporal-consistency/process/asset-migration/journal.md)；禁止借用其他用户环境或在/root安装项目依赖，详见[路径与环境规范](specs/项目规范/路径与环境规范.md)。

历史进展及失败过程完整保存在[进展报告目录](docs/进展报告/)和Git历史中；当前状态以上述接续入口及实际终态为准。
