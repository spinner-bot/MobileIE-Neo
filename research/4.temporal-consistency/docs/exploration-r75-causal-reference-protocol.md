# R75：发布因果RGB架构的自启动原生共同参考

登记：2026-11-13；Codex  (sqs_A100)。本文件为正式首步前冻结协议，不作活跃状态公告；实际执行以process/exploration-r75接续记录和各attempt的真实PID/progress/runtime为准。依据note57、0113、原生分块资格及两次丢弃资源试跑；不能把登记或功能资格当作已训练结果。

## 问题和角色

现有已发表共同参考分别为single-frame和future2窗口，缺少公开架构、无GT自启动、无未来、无需外部flow的RGB递归参考。用作者ReBotNet发布XS固定配置补这个信息条件，不设计新模块，不将普通适配称创新，不重开R74等旧STOP。该配置不是已验证的最终WACV主表S/M/L配置；没有作者权重/训练入口，结果只能标为**released ReBotNet-XS, task-trained self-start/tiled adaptation**。

64000步是本项目参考预算，不是原文500000×8GPU训练复现。原128训练的MobileIE/StableLLVE/FastDVDnet具有不同初始化、像素、优化器和训练计算；不以相同步数宣称公平收敛，不把本结果单独归因架构/容量，也不导入原文PSNR。用途是提供实际同开发数据评价条件下的参考点和后续有效方法必须超越的条件。

## 固定输入、预算与训练

- 作者固定提交dff81d3，原架构不改，XS890275注册参数；全部随机初始化，seed0/1各一个模型，均完整执行，不按首seed好坏取消另一个。
- 仅既有57 DAVIS train、3428登记帧和12 OLiVES train、384文件；三SYN开发视频及四ND开发场景排除。原最终test-dev30 JPEG保持未读，不获取新数据或将DEV反灌训练。
- 每步2 synthetic+2 real视频、各4连续帧，同位置原生384裁块，共16目标帧；不resize、无未来、GT只用于当前监督。正式位置从合法384范围均匀抽取，不能复用资源小试的夹紧128坐标。synthetic沿既有degrade/photons20或80/无JPEG定义；同seed/step输入身份与退化种子可记录复原。每视频共同水平/垂直翻转，不加新退化网格。
- 首帧history=该帧退化RGB，首帧也调用模型/计损失；随后history=自己的未夹紧预测，四帧完整BPTT，无GT history、无detach/teacher forcing或未登记warmup。部署同样传未夹紧完整拼接RGB，只在报告输出入口clamp0..1。
- seed各固定64000更新/0..63999 data-step；总128000正式更新、2048000目标训练帧（重复采样，非独立源图像数），不按墙钟截断、不延长或追加seed。Charbonnier mean sqrt(error²+1e-6)、fresh Adam4e-4/betas(.9,.999)/eps1e-8/wd0、cosine至1e-7、clip1、FP32/TF32关闭。发布架构本身的dropout/DropPath按train/eval切换保留。
- 顺序每次一条视频四帧反传、loss除4后累积四条再一步，等效batch4视频；不把四条图同时驻留GPU。每1000更新保存独立状态（model/Adam/scheduler/Torch RNG/data-step），每100记录训练流；仅最终64000进入共同DEV，不根据中途DEV选checkpoint。非有限失败不自动复跑/换损失；资源性退出须保留原终态，若恢复须准确记录checkpoint后接续范围，不伪称原进程正常。

## 部署和评价冻结

使用已通过人工CPU资格的`rebotnet_reference.py`：384 tile、256步幅、边界覆盖及短边replicate、正三角窗归一拼接；不缩放原生图、不用GT/未来/外部flow。完整拼接预测作为下帧history。28调用/1080p的21.976141824G仅Conv/Linear，另有状态/融合/其他算子，非手机时间。拼接改变边界上下文，正式输出必须实际全画幅和原生局部审阅，不声称整幅原架构等价。

最终冻结每seed的两个输出条件：`causal`部署递归，`current`相同权重每帧两个输入都为当前退化帧。后者是历史干预，非独立训练空间骨干；不能把它的下降当作新方法胜出。与既有R44 square_mse8000同seed原raw和strong recursive .5完整共同表对照，完整记录其不同训练预算。

先执行既有10输入条件（SYN六、ND四），两个输出×双seed=40新质量条件，与40既有raw/strong-rec合80共同条件/16域-seed均值；质量指标/GT flow接受规则沿既有评价，不为新方法改接纳域。JOD沿既有官方CVVDP配置、同完整原生帧/nominal30fps，40新感知条件，不能把不完整前缀当结论。预先固定全部10条件联系表，保留首/中/末完整FOV、128原生局部与误差；全20种seed/输入图实际查看，另存实际审阅记录，不回写原renderer标志。

双域双seed共四单元，每个同时核两组条件：对strong rec要求PSNR≥reference−.05dB、JOD≥reference−.02、Etemp≤.90reference；对same-weight current要求PSNR≥current−.05dB、JOD≥current−.02、Etemp≤.95current。保留完整八组True/False、逐序列最差值和所有原数据；须全部通过才把它列为有联合优势的进一步机制/手机成本候选。未通过也保留作为具体预算下的发表架构适配参考，不否定原论文或所有因果模型，不通过宽度/展开/LR/loss/长度/seed网格挽救。

两域四单元均通过后才按另登记固定接口扩到已取得REDS DEV与三手机资料，不能在当前协议写成已完成外部验证。当前纸稿不因登记新填成功主表。完整论文仍需新机制/信息证据、自然动态、手机与独立最终，R75本身不构成创新。

## 共享资源执行

原试跑PID4155198已退出，两次丢弃更新/八视频反传正常，0正式更新/0检查点；实际0号与预绑定一致。峰allocated678528000B、reserved710934528B、实际进程观测最高1242MiB，最少空闲37677MiB，同卡他人进程观测2026MiB。只证明此形态可承载，不证明长期吞吐或他人无减速。

正式任务每个整个子进程前绑定UUID，真实CPU数据/模型准备后重新11×2秒准入并持该卡项目锁，临CUDA前再核显存和实际PID/UUID；预计总增量4GiB/allocator2GiB，为既有任务至少8GiB增长+1GiB余量。相同形态可先在同组最多两卡各一个seed（总上限仍四卡），每卡独立新准入；未合格不启动。初始自身同步工作占空估计25%，每视频后主动让出至少0.1秒；这是比短试跑15%更高但仍保守的正式调度，必须持续记录。逐视频显存检查、定期实际卡负载采样；连续三次让出后≥90%或预留耗尽则在安全边界保存自己状态并退出/让出，不操作他人任务/驱动。新负载下无“卡永久已获准”假设。代码/数据资格/实际首步及状态记录完成后才可把接续入口状态改为已运行，冻结协议正文保持。

实现细化（正式首步之前）：逐视频通过只读NVML句柄查询绑定UUID的memory/utilization，避免长训练反复spawn CLI；初始完整准入和实际PID仍由nvidia-smi交叉核对。连续高负载判定相邻至少隔2秒，避免把同一个NVML采样窗口误当三次持续负载；内存检查每视频前后不降频。每100步记录累计资源最小值/最大值、工作/让出总量和最近16样本，训练窗口用固定SeedSequence和来源清单可复原，不把百万次遥测或重复长路径灌进Git。依据[NVIDIA只读查询文档](https://docs.nvidia.com/deploy/nvml-api/620/api/group__nvmlDeviceQueries.html)，不调用设备配置接口。资源中断若发生在四视频累积中间，保存最近已完成更新及该未完成step开始时Torch RNG，丢弃未提交梯度；恢复明确从同一个未提交data-step起，不重算已提交SGD更新。非有限/源码错误不自动重试。这些细化不改变输入、模型、目标、预算、门槛或旧STOP。
