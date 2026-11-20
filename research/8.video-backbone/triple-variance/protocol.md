# N1：三观测差分对原增强器随机波动的校准检验

2026-11-20；Codex  (sqs_A100)。先登记/实现/冻结，再首次运行。[基本恒等式与先例](../../5.prior-art-survey/docs/75-triple-difference-noise-identification-boundaries.md)。不是新统计理论，不训练/输出新增强方法，也不对旧滤波器调参。

唯一原F0seed0/1、4075参数/完整权重冻结。复用P1已固定四个DAVIS训练纹理/128裁剪及source SHA/坐标，不取新场景/DEV/最终。静止且无几何变换，不估flow；不能将结果外推到运动、真实ISP、自然夜间或独立最终。

独立参考：每纹理×每F0seed×photons20/80各128个独立原degrade（gamma2.2×.35、Poisson+σ.008、clip/uint8、无JPEG）观测，RGB输出clip[0,1]；FP64样本均值/无偏样本方差。共2048NN图，独立RNG SeedSequence([892,texture,photons,999])，不读取probe数据拟合参考。

待估计组按[Y0,Y1,Y2]的photon三种固定模式[20,20,20]、[20,80,80]、[80,20,20]，common_fraction=0/.5。每组64独立triplet、四纹理×两F0seed，9216NN图；合计11264，0优化/flow/GPU/DEV/最终。每triplet RNG SeedSequence([892,texture,pattern_index,int(100*fraction),triplet])，两F0seed使用相同噪声，不以其充当独立数据重复。

相关噪声构造明确保边际：基础暗强度l=(GT/255)^2.2×.35；common count~Poisson(fraction×min(photons)×l)，各帧再加独立Poisson(photons_i×l−common_rate)，除各自photons_i；read=√fraction×common N(0,.008²)+√(1−fraction)×independent N(0,.008²)。之后clip/uint8同原degrade。fraction不是异photon情况下的观测相关系数。每个triplet共享成分独立重抽，不是只抽一个固定pattern重复64次。

CPU2/OpenCV1/FP32 NN、FP64统计、torch deterministic；前向batch8（triplet的24图，每三张构成一组，不跨样本归一化）。逐条件固定预算，无墙钟截止。只存自有.cache的参考均值/方差、逐triplet池化估计、probe均值/协方差/期望差分图，不留全体RGB大数组；RNG和输入足以重建，未宣称逐bit重放。

可观测估计q=(Y0−Y1)(Y0−Y2)，不减未知GT或MC均值。先取中心[16:-16]、RGB均值，固定16×16不重叠池化得到6×6 signed估计；先pool再clip负值到0。MC参考V0同样中心/RGB/16池化。记录每triplet signed全局估计、负tile比例、正截断tile图的加权相对L1误差sum|estimate−reference|/sum(reference)。记录64triplet平均signed全局对reference比，以及positive平均图误差。0..255显示尺度只作图，不进估计。

主资格只看独立common_fraction=0全部24组（4纹理×2seed×3pattern），不得只选共同photon：signed均值全局方差比在[.85,1.15]；逐triplet局部正截断图相对L1误差中位≤.35；signed负tile比例均值≤.20，全部组均须通过。任一参考全局方差≤1e−8显式失败、不弃组。相关fraction=.5完整描述不作为必须等于V0的门槛，因为原假设不满足。

另保存probe样本协方差C01/C02/C12和独立参考均值差项，描述基本式中的相关与均值项量级；这些是有限样本解释，非逐字节闭合或整网噪声根因。原信息式无需GPU/优化资格，也不以代数自比较当性能通过。

全部24主组通过仅允许另登记实际运动条件下的有限校准，不允许直接GPU训练/调门/完整DEV/手机。失败停止固定三观测/16tile/截断/窗口/模式/相关构造局部网格，不叠平滑或训练补救。所有原STOP、最终封存/原84值/PDF020、7号及其他worktree保持；代码/来源/结果/过程/报告单独密集提交，阶段推送。
