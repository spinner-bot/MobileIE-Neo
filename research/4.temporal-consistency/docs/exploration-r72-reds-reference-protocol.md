# R72｜冻结参考的跨来源原生720p开发检查

登记：2026-11-13；作者：Codex  (sqs_A100)。R71固定资料获取、R70传统参考仍沿原进程运行。本轮不是新候选/训练，不能用结果重开旧目标/状态/损失网格；没有打开最终测试。

## 固定问题、资料与退化

只检查既有参考在新标准来源/原生分辨率下的迁移：同原MobileIE MSE/L1的取舍、强空间U-Net和五帧参考的差距是否仍出现。它不会隔离容量或证明新时间方法。REDS原任务不是LLVE；这里明确称REDS来源合成低光DEV，不能替代自然夜间配对/官方低光benchmark或独立最终结论。

必须R71原fetch001正常、256原PNG完整且固定四源联系表全部实际查看后，才准备本轮模型输入。源固定validation000/009/019/029各前64，RGB uint8原720×1280，不裁剪/缩放/亮度修正。两个光子条件20/80，完全复用exploration_r2_data.degrade(jpeg=False)：先sRGB数值幂2.2乘.35、Poisson再Gaussian .008、clip后uint8量化；它不是完整真实ISP或物理RAW复现。每源/光子一个固定PCG64流，seed为SHA256(`R72:{scene}:{photons}:0`)前8字节little-endian，不扩噪声seed。

GT256原帧、512低光观察/8条件；同源两噪声不当成两个独立视频。GT full-FB与有效域由原forward_backward每相邻帧估计、四源仅各算一次；公共评价沿原evaluate及八像素边界，不是光流真值。GT运动只供指标，不进入增强或post。原GT/flow/mask单独保存，两个退化共享；准备CPU2/OpenCV1，0CNN/SGD/GPU。全部源/输入清单正常后才执行模型评价。

## 仅最终冻结参考

两seed都固定已完整正常的唯一final8000，不看备份/选权重：R44 square_mse MobileIE4075、R69空间L1 MobileIE4075、R56 Nano696、R65 spatial_l1 U-Net4316259、R53 source-context FastDVDnet2479096。不重跑已止损commute20/状态/其他失败候选；这不是所有已发表方法穷尽比较或SOTA表。

保留各原模型RGB/BGR/FP32与输出clamp约定；Nano原乘法输出，U-Net原BGR归一，FastDVD原固定sigma15/255接口不以已知光子值校准。FastDVD使用原五帧反射边界window、最大未来2帧、原未缓存三次第一层和一次第二层；不增加five_current诊断或改原训练上下文。空间模型raw未来0；各空间raw另接自身raw的原half-FB/接受域recursive .5，首帧=自身raw首帧、无GT部署流。此处不追加FastDVD的post或新alpha网格。

共5模型×8条件×2seed=80增强CNN序列/5120中心，0SGD。每seed32空间raw+8五帧raw，再32自身recursive，共72输出；双seed144质量行/9216逐帧。Input另8行/512逐帧、0CNN，两者合计152质量行/9728逐帧。每模型先存全部FP32 raw并正常释放GPU，再由独立CPU2链计算公共质量/post，避免GPU等待长CPU评价；原始与post输出均保留自有cache。GT不填PSNR无限值的质量行，若感知时作控制另记录。

GPU只同组4–7，项目同时≤4卡；每seed一张。必须新外部两次空闲采样、项目锁，全部CPU模型/资料核验并准备首输入后、CUDA初始化前再两次新采样：无compute进程、≤64MiB、≤5%。卡忙只记录拒绝/等待新准入，不能抢占或由超时重启。GPU仅FP32/TF32off/cudnn deterministic，Torch2/OpenCV1；同轮所有神经参考同后端，不将与旧CPU集合的微小差异解释成方法提升或做逐bit复现。每帧有限/维度/范围检查，科学失败原样保存、不自动重试/换精度/模型。

## 完整后续与范围

两seed全部神经与CPU终态正常后才汇总全部条件与38个条件均值（9输出×2seed×2光子，另Input2）；4原视频为统计单位，不将512帧当独立样本、不只报跨seed有利均值。

感知另用既有ColorVideoVDP0.5.7/standard_fhd/replicate、原生720p SDR，**来源24fps**，不回写旧集合名义30条件；相同固定显示假设，非手机/人类MOS/单独闪烁评分。仅完整缓存后新空闲GPU准入：144模型输出+8Input+4唯一GT控制=156行/9984比较帧；不为两个噪声重复GT控制，不重跑增强CNN。跨数据集不同尺寸/帧率的绝对JOD不当统一排名。

固定4源×2光子×2seed=16联系表，时刻0/32/63，全FOV缩图+原生中心128；11列Input/GT、MSE raw/rec、L1 raw/rec、Nano raw/rec、U-Net raw/rec、FastDVD window，全列不选片/曝光/颜色修正。生成不等于实际查看，全部实际查看另记，无人工全视频/MOS宣称。完整后才写论文来源独立表；新手机/自然夜间/最终测试不打开。

公开预训练资料可能重叠须保留：本项目适配只MixedPool既有训练源，没用REDS；这不能保证全部公开起点未见REDS。逐模型保存已核验来源/声明与未核实边界，不能用“项目没训练过”改称完全未见。FastDVD等原资料/目标/训练计算不同，不能据此纯架构/容量排序。传统块匹配R70在另一集合仍未结束，其部分分数不进入本轮；本轮传统强对照是固定自身输出rec。

数据/神经/CPU/感知/汇总/联系表接续代码及来源冻结完成并提交后，才按上述顺序启动；无墙钟截止/科学自动重试。代码、配置/准入、完整结果、过程、解释、报告分别密集提交、阶段SSH推送/no-ff；资产只zrh自有data/project-python，双空格作者，不改移动演示。论文新方法、可信联合收益、标准自然动态配对、完整手机成本和独立最终仍未完成，目标active。
