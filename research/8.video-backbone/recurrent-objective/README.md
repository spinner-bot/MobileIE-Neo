# T2：直接监督部署递归的完整已知控制

2026-11-20；Codex  (sqs_A100)。**已完整负结项，所有原任务终态/GPU释放，不重启原链。** [完整决定](decision.md)、[0184](../../../docs/进展报告/0184-T-26112021-部署递归监督完整评价与止损.md)、[真实终态](process/terminal001.json)、[全部过程](process/journal.md)。论文goal仍active，无确立创新/手机/独立最终证据。

保留同seed原4075参数F0完整权重，独立clip_gt与rec_gt两模型各4000×两seed，共16000正式更新、四唯一最终；零更新原生资格32NN/24DIS单列、CPU4/CUDA4优化资格全丢弃。[原协议](training-protocol.md)和47全链源/输入、24原生资格源保持。rec训练/部署保留原DIS/.5/原接受与RGB历史，窗内RGB BPTT、field/mask stop-gradient，普通递归训练已有直接先例，不主张创新或推理零额外开销。

[全360条件/72均值](report/joint/tables.md)包含120新质量与官方JOD各5376帧、复用T1全部240。恢复0/4、A成对0/44、B10/44、额外5%学习0/8，完整A/B与成本扩展False。对clip_gt DIS，SYN PSNR+.10–.11dB、14/20条件PSNR更好，但全部20条件Etemp更高、四组+1.50%–2.46%、联合0/20；raw对raw也全部20条件时间更高。保留正例/取舍，不能用训练梯度非零代替收益。

[24对/480条件/96均值差分](report/analysis/differences.md)、[四固定图实际审阅](report/contacts/actual-review001.json)、[72点PNG/PDF/CSV及实际PNG审阅](report/plots/actual-review001.json)均保留。PDF仅导出未逐页审阅，图非播放/MOS；重复DEV/多退化/2seed不是独立视频/总体显著性。参数/MAC和GPU训练占用不当手机速度。

按原登记停止固定原F0/native DIS/.5/四帧递归监督的alpha/mask/插值/loss/LR/长度/seed局部网格，不扩失败成本/外部/手机，也不继续向T1/T2加损失作诊断循环。后续需实质不同的信息/结构问题、先核先例再有限登记，不能预先宣布另一架构有效。旧STOP/原84值/snapshot003/PDF020/最终封存、7号与其他worktree保持。

原评分21:38:05、协调21:38:33正常完成，全部原PID实际已无；历史静态state不作重启依据。过程实际按测速长sleep20/15/20/25/8分钟、醒后核同句柄，没有墙钟截止。源码/结果/图审/分析/决定/终态/过程/报告分别密集提交及阶段SSH推送，资产全在zrh自有data，指定作者/no-ff保持。
