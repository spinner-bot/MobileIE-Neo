# B1：原模型内部的外观索引历史，零更新必要检验

登记：2026-11-21；Codex  (sqs_A100)。原因：M1恢复失败停止；不再改变原空间骨干或随机重训，改问紧凑外观状态是否提供有用历史。直接VPN/曲线/LUT先例见[note77](../../5.prior-art-survey/docs/77-appearance-space-state-and-existing-propagation.md)，普通双边传播非创新。本协议先于B1实现及任何新数值执行。

## 单一固定算子

两seed原R44 square_mse的4075参数F0完整权重/eval/冻结，head.block1当前输出z是平方前的12通道，之后原head FSTS、body、两门、tail全部不改。当前低光x先replicate-pad2，再5×5均值、clamp[0,1]、逐通道开方，得到三维键q；不是亮GT或另一teacher。17³规则网格上以三线性八角权重S_q进行splat/slice，无额外网格卷积、位置/时间坐标、flow、学习descriptor或GT提示。

每帧构建n_t=S_q z/(HW)、d_t=S_q 1/(HW)，形状4913×12和4913×1。历史状态N/D从首帧复制初始化；之后**先读旧表、后更新**：a=(slice_q N)/(slice_q D)，质量小于等于1e−8的查询回退z，融合z'=.5z+.5a，接原全部非线性读出。每帧结束N←.5N+.5n_t、D←.5D+.5d_t，状态只存原当前z的统计，不把融合z'再写回。每窗口重置，首帧无历史直接原z。FP32紧凑复制持久状态13×4913×4=255476字节，和分辨率无关；不计模型/当前临时表/guide/索引/工作区，不称峰值RAM或手机速度。大运动可按颜色跨位置查询，但同色不同物体/纹理会混合，不保证真实对应、均值不偏或已去噪。

固定五输出：raw；原raw自身half-FB .5递归；原raw自身native DIS ULTRAFAST .5递归；history_bank；current_bank。current_bank使用**本帧**n/d作同样slice和.5融合，首帧也保持raw；从第二帧起它是同表非局部空间平滑控制，永不保存过去。history_bank保存其独立N/D，不能读取当前表后才输出或共享其他模式状态。两个经典参考按原RecursivePipeline/独立自身融合RGB计算，不借GT部署场；这不是另调FB/DIS/alpha。

## 数据、预算、资格与评价

只CPU/CUDA_VISIBLE_DEVICES=''，Torch intra2/inter1、OpenCV1、FP32。现有MixedPool seed0/1、step8000/8001，B4T4/128，2SYN/2ND，合16窗口/64帧，训练场景而非独立DEV。两seedF0固定、不新增SGD/teacher、0GPU/新DEV/最终。原head当前编码共享64张；为资格另外完整原net调用64张，分解原续接64张、history续接64张、current续接64张，共128完整head前缀图和256后半图，不能把共享实验调度当成独立方法成本。

一次执行内先人工常值小表检查质量归一、所有查询权重和1、常量value恢复、首帧/空表回退、只读旧表/更新顺序、实际独立紧凑storage与255476B。人工表无CNN/SGD，检查调用单列。对全部64真实帧，原分解续接对直接net归一RGB最大差≤2e−6；首帧候选/空间控制对原分解严格一致，全部有限，旧权重前后不变。失败保留、不放宽阈值或自动重试科学参数。

每窗口统一GT估计half-FB+8px边界得到评价域，各方法共用，但部署不接GT。各经典参考自身raw上独立场，48对FB部署、48对DIS部署、48对GT评价，共144双向对；DIS96calc、FB部署与GT合192calc。统一显示clip到[0,1]转0..255；沿原PSNR/SSIM/edge/GT残差时间误差，评价输出必须有限，无空接受域跳样。80完整序列行/320逐帧，每域seed每输出4窗口，共20均值。保存全部窗口/帧/预测与键质量接纳率、状态和raw范围。训练裁块/4帧不能替代完整长视频、官方JOD、自然夜间、手机或独立最终。

## 继续门与停止线

history_bank相对raw/current_bank分别四域seed均须平均PSNR≥−.05dB、每个最差窗口≥−.50dB、平均Etemp≤.95参考；对两个经典参考分别四组须同质量保护、Etemp≤1.05参考。全部16比较同时通过才允许**另登记**完整视频/学习设计和真实成本；这只是必要信号，不能自动算论文创新。所有局部正例/SSIM/edge/接纳与启动效应保留，不据部分结果挑seed或表分辨率。

任一完整门失败，停止此17³/5×5开方RGB键/head平方前12通道/八角插值/.5融合与decay/质量阈值组合的grid/guide/节点/alpha/decay/阈值/seed/训练补救局部搜索。不得据此扩大DEV/手机或重开R37/R73/R80/R84/R87/R88/V1/M1等已停网格；不作所有双边/检索/非局部方法不可能性结论。若通过也须正面证明区别于VPN/历史LUT和强空间平滑的贡献。

仅冻结具名源/固定参考，不做完整transitive依赖或Adam逐bit工作；代码/源清单、完整结果、图与实际审阅、决定/过程/报告分别按文件提交并阶段SSH推送。所有运行资产只在zrh自有data目录；main/作者/no-ff/旧STOP/封存最终/7号和其他worktree边界保持，论文goal active。
