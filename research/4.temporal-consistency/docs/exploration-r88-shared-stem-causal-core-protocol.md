# R88：更早共享stem的几何分支与完整因果视频核心

2026-11-17；Codex  (sqs_A100)。登记先于新图像/训练/求解资格。依据note71和R87待核问题。普通共享encoder、合成对应监督、正交描述和LK均有先例，不主张创新已成立。本轮目标是完整因果视频链及同容量推理形状对照，不再单独做一个小度量检索实验；R84/R87的固定z/hard matcher/尺度网格与全部旧STOP不解封。

## 架构和唯一固定对应算子

每seed冻结原R44 square_mse4075参数、eval/无grad，原RGB路径不改变。共享head第一个Conv5/PReLU输出A（12通道），增一个无bias Conv3 stride4/pad1、输出4通道，432新增权重、总4507。描述减每图每通道空间均值再通道L2归一化eps1e-6。原重建与新分支隔离；RGB保持不是正交约束造成的。解析新增MAC约27HW、1080p .056 GMAC，不是完整实测。

两条件plain和complement：plain直接学4×108 kernel；complement在固定恢复head最后Conv3权重W(12×108)的ker(W)基N中学4×(108-r)系数、kernel=V N^T。CPU FP64 SVD、rank阈值smax*1e-6，N转FP32固定；两个head初始有效kernel完全相同（seed8800+seed，先采V/sqrt(108-r)，plain复制complement的FP32有效kernel）。训练自由度不同、最终部署均fold为432权重，不称严格容量/优化等价。W不包括bias，权重行空间正交不等于输入图像真nullspace、统计独立或逐点描述夹角正交。

固定三层quarter特征金字塔（avg2/ceil、每层再L2归一化），粗到细，每层5次稠密局部平移Gauss–Newton型更新。中央差分求previous梯度、按当前flow采样previous及梯度；5×5窗口平均累计H/b、阻尼λ=.001*trace(H)+.0001、det clamp1e-12，增量逐轴clip[-1,1]，每层位移界对应native±64。每次更新均执行，不作迭代/窗口/层数/阻尼网格；邻域flow近似非严格inverse-compositional LK，不宣称优化单调/DeepLK复现。层间flow上采样×2、裁回实际奇数尺寸。quarter坐标中心是native0,4,8...；native场按x/4,y/4映射采样并×4，不用错位的resize假装精确坐标。

部署估两方向、沿原native FB阈值1+.05*(两流范数和)/bounds接纳；自身上一fused用原warp/alpha.5融合、无效回当前，首帧严格F0。只持有上一quarter4通道描述及native RGB fused，理论状态3HW+4ceil(H/4)ceil(W/4) FP32；workspace/场/当前激活另计，不称峰值RAM。没有外部flow/GT/未来或学习confidence。true/zero两路分别维护自己的RGB历史，zero使用同一true接受mask而位移为0，仅干预warp，不另估mask；当前映射恒为F0 alias，不是新训练空间模型。

## 数据、训练与预算

每seed原MixedPool data_step8000..9999，B4/T4×128、2DAVIS/2OLiVES，只取四GT第0帧生成已知几何对，原loader仍构造/预载原cache。两来源均重新合成噪声，不是实拍噪声监督。每b native平移分量从[-192,192]/16均匀采样（在OpenCV1/32相位上精确），previous/current clean由原warp生成，分别独立degrade(photon20或80、noJPEG)，同对共享photon。独立SeedSequence[88,seed,step,b,2]定几何/photon，side0/1定噪声，两head共享全部输入。quarter真位移为native/4，没有GT flow估计教师。

每head损失固定为：5×5 query格{8,12,16,20,24}²、9×9候选的余弦/.1 softmax CE（已知位移的四邻双线性概率标签），加最终quarter flow对已知平移的中心去8px MSE/16。只训练描述权重，原stem/F0无grad；CE辅助避免无纹理/常量塌缩，不主张该损失组合新颖。Adam1e-3/(.9,.999)/eps1e-8/wd0/globalclip1，FP32/TF32off/确定性。

两seed各两head×2000新更新，共8000 descriptor optimizer更新，0原骨干更新。每seed每步共享一次B8 stem，合4000 B8/32000 stem图；两head各B8描述合64000图；每head每步B4单向solver，合32000单向训练场。原完整F0训练前向0，不能写成0CNN。500步备份，唯一最终2000，不看DEV/挑checkpoint/改预算/加seed，无墙钟截止；仅真实resource pause可从完整双head更新边界恢复优化器/步数，不因观察超时重启或数值失败重试。

## 实现资格与正式启动

CPU先固定人工4通道32×40平滑随机特征、quarter已知位移(1,-.5)，中心去8后中位EPE≤.5quarter、全部有限；无阈值放宽/换纹理。实际每seed第一个注册batch：两head初始描述差≤1e-6；原/shared-forward RGB差≤1e-6，两个head各2丢弃更新（两seed共8），梯度有限且norm>0、冻结F0始终无grad；fold descriptor max差≤1e-5、complement归一化W交叉残差≤1e-5、4507部署参数/首帧/紧凑状态检查。预算每后端16 full-F0图+16 stem图，8丢弃更新；不继承资格权重。CPU全部过才同样CUDA资格，之后才能正式训练。

使用现有整进程UUID预绑定、CPU模型/全部资料/首batch真实准备后重新连续采样/锁/实际PID UUID、4GiB估计/2GiB allocator/8GiB既有增长保留+1GiB安全、25%让步/逐batch NVML监测，优先空0/1卡、同0–3组最多四张，允许符合余量的共卡但不动他人进程/时钟。CUDA不合格或数值失败保留停止。任何resource恢复只续原完成边界，算法预算不变。

## 最终完整共同评价、感知与解释

最终两seed各plain/complement true+zero、原六SYN64与四ND16，80新质量/3584逐帧；复用原F0/half-FB/DIS的60质量/JOD，合140条件/28均值。所有source/seed保留，原GT参考flow/mask/8px边界不改，不筛accept区域。原raw缓存可节约评价完整F0，但每帧仍真实共享stem一次（两head共用），合896 stem图/1792描述图；每head一次双向场供true/zero共用，合1752双向对/3504单向场，不能说0CNN或部署无骨干。

全部80官方ColorVideoVDP0.5.7/standard_fhd/名义30fps/replicate、3584比较帧；原adapter和共享GPU规范不改。固定四联系图：两seed各bear_photons20和video_47，t0/中/末，Input/GT/raw/half-FB/DIS/plain_true/complement_true，实际查看另记；这是预定样例、非全部源目测或MOS/视频播放。原指标完整覆盖所有源。

每域-seed逐一报告plain/complement true对F0及两经典参考、各自zero。主候选complement：PSNR/JOD对每参照分别≥-.05dB/-.02，最差逐条件分别≥-.30/-.10；对同权重zero及独立plain_true需Etemp≤.95，各四组。A要求相对两个经典Etemp≤.90，B要求≤1.05；A/B只属不同质量—成本取舍，不回写旧失败或称B达成10%时间改善。普通plain与zero全部保留，若只有plain有益不得归因正交补或称新贡献。

只有complement任一完整A/B质量资格过，才登记/执行同轮完整CPU成本，与原half-FB和DIS比较，各两seed十条件前16帧/每模式2warmup+3正式，3pipeline共4800真实F0、120 warmup/180正式序列，加入head/全部solver/转换/warp/sink；A还需cost≤较快经典*1.05，B需≤*.80。未过不扩失败候选成本/外部/手机。任何结论均为反复DEV，未建立新颖性、自然动态/手机或独立最终。

依赖按CPU资格、CUDA/正式、评价阶段分别在执行前冻结，不改活跃源；实现/原资格/全部checkpoint前缀/原终态/结果/解释/过程/报告分文件小步提交并阶段推送，自有data/project-python/main/指定作者/no-ff。原全部STOP/保留最终/paper020和完整论文目标active保持。
