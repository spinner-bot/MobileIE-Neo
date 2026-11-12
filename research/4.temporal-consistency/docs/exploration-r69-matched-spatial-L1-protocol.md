# R69｜同原骨干的空间 L1 必要对照

登记：2026-11-13T04:34:14.980863+08:00；作者：Codex  (sqs_A100)。登记时无R69资格、代码执行、正式更新、质量或GPU任务。R65/R66/R68均完整，固定commute20与全部旧失败候选保持止损。

本轮只补一个可进入共同论文表的基础对照。R65空间U-Net的PSNR/JOD有信号，但它使用L1，现有强MobileIE使用MSE；结构、预训练等混杂已披露。本轮在原MobileIE上只改空间目标为L1，不由交叉排名断言容量不足，不做新时间结构或宣称L1创新。R26旧小资料的Charbonnier/MSE不能替代本轮57DAVIS+12ND训练资料、原完整MBR/BN、B4T4crop128和最终8000的L1条件。

## 固定唯一训练条件

原R41 create_model(seed)从同官方slim函数恢复完整expanded MBR；训练49978总/47239活动参数，冻结无用warm头，head/body均原平方FST、原注意力/卷积/BN。没有R44 rational/R64 state/递归训练路径或新输入。最终fold为原4075参数RGB单帧模型，raw推理计算/无历史条件与原MobileIE一致；这不保证质量或手机延迟。

seed0/1各8000正式更新，共16000，从原公开初始化重置、不从R65/旧最终8000继续。仅全部原始未clip输出对GT的四帧等权L1。原MixedPool.batch(seed,data_step)0..7999，B4T4crop128、固定50/50 SYN/real、原量化/退化/裁块/翻转完全不改，四次B4 BN调用。Adam1e-4/betas(.9,.999)/eps1e-8/wd0/global clip1，4000后重建Adam，对齐既有R41→R44的两个4000阶段；FP32、TF32off、cudnn deterministic、Torch2/OpenCV1。L1与MSE梯度尺度/clip耦合仍不同，比较的是完整目标设置，不声称仅改变鲁棒性。

MSE使用已有正常R41 expanded前4000→R44 square_mse后4000的同seed8000参考，不重复训练/逐bit复现。先核旧配置/窗口范围/正常终态；如实际条件不符，保存差异并停止声称严格单因素，不自动重跑一套MSE救对照。记录本轮完整窗口，对已有R65相同seed的0..7999窗口身份作普通结构化一致性检查，不比Adam张量、跨GPU输出字节。

每1000保存备份/Adam与唯一final8000导出；不以DEV挑备份。没有目标混合、权重、crop、比例、LR、长度、seed网格；无论输赢均不以本轮扩局部网格或重开旧冻结族。

## 资格、资源与完整后续

先CPU一个实际seed0/data_step0窗口：同起点L1/MSE初始预测一致、损失/活动参数梯度有限、首尾读出及FST实际变化；两臂各1次丢弃SGD，不继承正式训练。原初始及更新后eval/fold同真实窗最大差≤1e-4；不是对任意输入或旧R63人工失败的改判。CPU失败保存停止，不能宽阈值重试。

GPU仅同组4–7、每seed一张且总≤4。新外部两次空闲采样/锁，完整CPU资料和首窗准备后、CUDA初始化前内部两次新采样：无compute进程/≤64MiB/≤5%。每seed只L1作1次丢弃CUDA资格，再重新载入原初始模型/Adam；丢弃更新共CPU2+CUDA2，正式16000分开。无墙钟截止，科学失败不自动重试，活跃依赖冻结。正式启动前必须具备完整最终评价/协调接续代码，避免无人值守丢结果；当前尚未启动GPU。

全部正常且两个final8000 fold过后共同CPU2/OpenCV1：原六SYN64（3源×2条件）、四ND16，raw/自身raw recursive .5，两seed共40质量行/1792逐帧、20CNN序列/896中心。全部预测缓存与逐条件保留，并完整列同seedR44 MSE、R65 L1、旧强已发表参考；只评价最终点。经典rec的运动/历史开销与raw分开，0未来/无GT部署运动。R66原固定ColorVideoVDP0.5.7/standard_fhd/nominal30fps/replicate在完整缓存后再新GPU准入补40行；非手机/人类MOS，旧门槛不改。

固定全部10源×2seed=20图，原0/mid/last时刻，全FOV缩图+原生中心128，无颜色/曝光修正，Input/GT、MSE raw/rec、L1 raw/rec、R65 L1 raw/rec全列；全部实际看后另写actual review。没有新手机/Night/DRV/最终测试或重算卷积预算。若L1改变质量/感知/时序取舍，保留两种目标强参考，不把某域均值当联合胜出；若无益，同样止损。本轮是基础对照，不足以建立方法贡献。

代码、资格、来源/配置、完整终态/表图、过程、解释和进展报告逐文件小步提交，阶段fetch核分歧后SSH推送、merge/pull显式--no-ff；指定作者、自有zrh data/project-python保持，不改移动演示。论文创新、可信联合收益/标准自然动态/手机与独立最终仍未完成，目标active。
