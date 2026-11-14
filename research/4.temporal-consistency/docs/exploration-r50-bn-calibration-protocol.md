# R50｜冻结权重、训练资料BN重算的单一控制

事前登记：2026-11-12T18:43:26.795951+08:00；作者：Codex  (sqs_A100)。R48已完整失败并冻结，R47仍在最终CPU链；本轮不是重开α/loss/LR/长度/seed网格。依据note39及R48未校正raw的事后DC误差证据，检验空间强基线对BN统计的敏感性。全部已知控制，不是论文方法创新。

## 资料、状态、唯一预算

六个完整MBR训练态：R44 square_mse最终8000、R48 plain与joint最终12000，各seed0/1。保留原全部权重/BN affine与架构；不做SWA、优化器或反向。原checkpoint只读，比较校准前后所有named_parameters须完全相等，state_dict改变只允许BN running_mean/running_var/num_batches_tracked。使用官方torch.optim.swa_utils.update_bn，冻结参数、eval起点，经包装保持每窗口四次B4调用。完整训练资料MixedPool，固定共同seed=50、data_step0–127，128窗口B4T4×128，共2048输入crop图像/模型、512次BN调用/活动层。六模型完全同输入顺序；没有GT参与统计、DEV或TEST前向参与适配。累计平均各批统计，不声称精确整体协方差。未调用warm层count=0，其它count=512。

固定后端GPU FP32/TF32off、Torch2/OpenCV1、确定性；先CPU小型解析BN均值/批次方差核验和一个完整原模型的两窗口4×B4×128丢弃校准，fold≤1e−4/权重不变/只BN buffers变。没有丢弃SGD。若GPU4–7组有空闲，外部gpu_guard双采样、完整CPU源后内部再次双采样均无进程/≤64MiB/≤5%才启动，一张卡依次六模型，保留R47链；无空闲不抢占。GPU校准首次模型也检查权重和fold再继续。源与过程逐文件入库，没有墙钟截止/数值自动重试。

## 固定评价和判据

仅一次最终校准权重；fold后仍4075，与完整MBR eval差≤1e−4。统一CPU评价R30六SYN×64与既有四ND×16。三个家族、双seed，每个raw和原self-recursive .5两读出，共120质量行、60新CNN序列、5376逐帧行；自身raw估流、首帧/拒绝回退，GT不参与部署流。不做finite、alpha网格或手机/最终测试。存全部raw及逐帧mean-channel误差MSE分解；相同定义原始raw来自R44/R48已完成输出。公开所有家族/域/seed，不能只挑video_121改善。

主要机制描述为校准前后ND四源平均绝对双seed PSNR差及DC误差变化；两个seed不支撑方差/显著性结论，变化说明统计影响，不证明此前偏差全部来自BN。

每家族单独的“可保留强参考”门槛：raw两域的双seed平均ΔPSNR均≥−.05dB；所有seed/domain≥−.20dB、所有单条件≥−.50dB、各seed/domain时间比≤1.02；ND四源平均绝对seed差缩小至少25%；且至少一域双seed平均ΔPSNR≥+.20dB。递归结果完整报告作影响检查，不将其收益改写为R48成功；若raw通过，递归也需同一质量下限及时间比≤1.02才称对该强递归参考可采用。没有上述门槛也完整保留诊断，停止此128窗口累计统计策略，不扩窗口数/momentum/域比例/seed网格。

不会把校准当新时间方法、0新增部署参数当0训练成本或改善当独立未见泛化；所有开发源已经反复使用。论文创新/外部自然夜间/手机/最终验证仍未完成。所有运行资产只在zrh自有项目data，结束后代码/结果/决策/报告分别提交并阶段推送。
