# R64 原dense骨干与内部历史共同适配：固定三组任务对照

登记：2026-11-13T01:16:51.120318+08:00；作者：Codex  (sqs_A100)。旧R63人工压力资格未通过，原结果/门槛不改；单次定位仍有有效RGB差。本轮独立检验实际训练窗/任务的适用性，先CPU/实际CUDA资格，全部通过后才启动正式训练。不是把普通R33模块改名为创新，不重开已冻结各轮局部网格。

## 原问题与模型

唯一问题：R34空间dense→DW替换、R37骨干冻结、R48输出递归未共同隔离的“保留强dense空间函数，原骨干与内部历史共同学习”能否达到强post的画质—时序质量，并去掉外部flow。直接复用dense_joint_state_r63.py / R33原路径，零readout初始化、全部原MBR可训练而tail_warm冻结；固定8通道/下采样4/±2低分辨率offset，无结构/LR/尺度/长度网格。当前未建立新方法/创新。

每seed从R44 square_mse最终8000更新的原expanded完整权重起点，三组plain/spatial/causal共同初始化；spatial真实执行同容量支路，以current当previous，非跳过容量。plain4075部署，spatial/causal4618=4075+543；后两者空间MBR fold后状态路径仍在，非单帧零额外计算。causal同权重current另为诊断，不代替独立训练spatial。真实状态8×ceil(H/4)×ceil(W/4)FP32，首帧=current；窗口内BPTT不detach，推理仅过去，未来0。

## 资料、优化与预算

同既有MixedPool，57 DAVIS训练源3428帧+12配对实拍训练源，排除既有所有DEV/最终源。两个seed各三模型×4000新更新，共24000；每模型总计12000。B4T4×128、共同50/50窗口和翻转，data_step8000..11999，R48一致。每帧B4 BN调用，共4次/更新；相同raw clip[0,1]后四帧MSE，含首帧等权，无额外时间loss/flow/BN校准。三组均fresh Adam lr1e-4/betas(.9,.999)/eps1e-8/wd0、全局clip1，FP32/TF32关闭。固定最终4000新增权重，1000间隔仅备份不挑质量。无墙钟截止，无预算增加/追加seed/自动失败重试。

## 任务资格（与原人工压力检查分开）

先CPU2/OpenCV1仅读取每seed固定step8000的一个既有训练窗；不打开DEV/最终图像，权重/数据来源入清单。三组train/eval初始raw差≤1e-5。每组两次丢弃SGD lr1e-6/globalclip1，共12 CPU丢弃；从新源初始化正式训练，不继承这些更新/BN/优化器。记录真实clipped loss、原FST/卷积/readout非零梯度和有限性、所有活动参数梯度存在/有限；初始readout使部分新梯度0可接受，记录两步后实际field/mix/project梯度，不虚称全部立即可辨识。causal丢弃更新后的末帧对过去输入梯度须>0，spatial为0。每组初始和两次丢弃后eval原始输出fold差都≤1e-4，部署参数数严格4075/4618。任何一项失败原样保存停止，不放宽容差/切精度/形状网格。

CPU通过后，若有空闲GPU，以相同固定真实窗/丢弃预算验证CUDA；同组4–7最多四张，外部gpu_guard新双采样无计算进程/≤64MiB/≤5%，CPU完整资料/首窗加载后内部再次采样，全程持有项目锁（不声称是其他用户预约）。卡忙只保存拒绝或等待新准入，不抢卡。两个seed正式使用同一GPU FP32后端；GPU门槛未通过不启动任一正式训练。所有CUDA依赖源码活跃期间冻结。

正式训练保存每模型1000步训练/部署checkpoint、来源与fold marker，最终raw fold≤1e-4才能自动评价。任务窗资格不能证明任意输入全局数值等价，R63压力失败仍须披露；最终CPU评价也须检查有限性，不以部署通过回写旧gate。数值失败停止并保留，未来是否重审要独立依据。

## 唯一最终共同评价与止损

仅既有R30六SYN64帧（3视频）+R10四native_crop ND16帧；两个seedCPU2/OpenCV1。每模型raw+自身raw经典recursive .5，另causal同权重current raw，共每seed70质量行/40CNN序列/1792中心，双seed140行/80CNN/3584中心；每seed逐帧3136、双seed6272（7输出×448）。公共GT评价流/接受域使用既有缓存，部署post只用自身raw的half-FB，不能把GT用于部署。完整所有输出数组仅自有cache，全部源/seed保留。经典finite历史参考已有，不新增冗余CNN。

两个主要质量门槛只看**causal raw**，同时对新plain recursive、新spatial recursive、旧R44 square_mse recursive；每域每seed必须逐一满足PSNR≥参考−.05dB，逐条件最差PSNR≥参考−.5dB。A额外要求Etemp≤参考*.90；B为有资源取舍的质量资格，要求Etemp≤参考*1.02，且causal raw对同权重current每域每seed Etemp≤*.95/PSNR≥−.05。B不是降低闪烁10%的成功，不覆盖A失败。A也必须有对同权重current的≤*.95时序贡献及PSNR≥−.05。全部条件不容只取跨seed平均。

完整保留全部raw/recursive、旧R44/旧R48 plain与R45 RGB强参考对照，报告条件/训练预算差；causal recursive只是完整参考，不能替代raw通过上述去外部flow问题。以上均为开发筛查，非主会创新/最终结论。若A/B任一质量资格通过，才登记完整CPU流水线（含原经典flow/warp）重复计时；资源B仍须完整CPU中位耗时≤各对应经典递归参考*.80，不能用参数/MAC/无flow推断手机时延，手机需后续真实硬件。实际A还须不超过参考1.05成本才可称联合收益。

若A/B均False，冻结本固定dense joint路径的宽度/尺度/offset/loss/LR/长度/追加seed，不扩大局部搜索，记录依赖收益/失败限度。即使有信号，通用模块组合/共同训练本身不是创新，需另行先例/机制/效率/外部验证，当前不打开独立最终测试。R60/手机无GT不得用于调参或选checkpoint，移动演示不动。

所有大体量运行资产留/data/Users/zrh/data/MobileIE-Neo，使用project-python；源码、资格、训练/评价结果、解释、过程、报告分别小步提交。阶段fetch后SSH推送，有合并显式--no-ff，不rebase/squash/force。作者Codex  (sqs_A100)。
