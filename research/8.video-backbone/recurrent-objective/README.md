# T2：直接监督部署递归的已知控制

2026-11-20；Codex  (sqs_A100)。当前已实现完整训练/质量/JOD/全表/固定图/协调链，正式固定预算运行中，**尚无完整新DEV或效果结论**。[0181](../../../docs/进展报告/0181-T-26112020-部署递归监督原生资格与固定完整训练启动.md)、[完整协议](training-protocol.md)、[实际启动前缀](process/observations/checkpoint001/status.json)、[过程](process/journal.md)。

[先例及区别](../../5.prior-art-survey/docs/72-deployed-recurrence-and-supervision-entry.md)：递归最终监督已有先例，不主张创新。保留同seed原4075参数F0完整权重，用独立clip_gt对照rec_gt（原DIS/.5/原接纳、RGB历史BPTT、field/mask stop-gradient）。推理保留真实DIS/warp/历史成本，参数/MAC不当手机速度；这不是给T1换教师或延长KD。

[零更新原生资格](preflight-decision.md)正常27.650654秒，32NN/24DIS/0SGD/DEV/GPU，前向native最大差1.7881393e-7、过去梯度rec非零/逐帧0，只证实现。随后[冻47全链源](process/training-source-freeze001.json)，CPU4/CUDA4优化资格正常且全部丢弃，再从原F0重启正式预算。原协调3246208/start_ticks111602758、seed0原worker3247554/start_ticks111620942；20:01:51每模型63、126正式更新，seed1为0，活日志继续，不能据本静态数值重启。

固定两模型各4000×两seed16000，唯一最终全120质量与官方JOD各5376帧；复用T1全部240合360条件/72均值/18输出，44强post主要比较、8额外学习增益比较及恢复保护、四固定图和完整成本止损全部先登记。完整A/B质量任一通过才另登记成本，失败关闭本固定配置局部网格，不扩失败外部/手机。所有通过也只是较强已知优化参照，创新仍待建立。

长等待按实测约.58秒/共同步估算sleep，醒后核实际PID/start/runtime；自动链没有墙钟截止，资源75保存完整共同步状态接续，科学失败不自动重试。所有旧STOP/T1/V1、原84值/snapshot003/PDF020/独立最终封存、7号与其他worktree保持。自有data/指定作者/no-ff/小步分别提交及阶段SSH推送，论文goal active。
