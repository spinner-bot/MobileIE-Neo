# P1：平方前粗能量历史的零更新可证伪筛查

2026-11-20；Codex  (sqs_A100)。先登记、实现和冻结，再首次运行。直接先例见survey note73；幅度/相位分解及时间滤幅度已知，本试验不宣布创新。旧V1/T1/T2及R51/R73等STOP全部保持，不开旧网格。

问题：在原4075参数完整F0首次FSTS前，历史只影响粗尺度能量、当前有符号载波保留，是否比空间处理和同尺寸有符号特征历史更有效？不估计flow，不存RGB，不更新权重。中心差分不是经过带宽校准的Riesz正交对，不声称精确相位不变或物理噪声估计。

设原head.block1输出u，m为replicate边界3×3均值，b=u−m。Dx/Dy为replicate边界中心差分，p=b²+Dx(b)²+Dy(b)²；q为4×4不重叠均值池化p。状态s首帧q，后续s=.5q+.5s_prev。g=sqrt(clamp((s+1e−6)/(q+1e−6),.5,2))；bilinear、align_corners=False上采样g，u'=u+(g−1)b。其余首次FSTS乘法顺序、body、HDPA、tail完全复用。粗能量状态detach且独立clone；新参数0，原4075冻结。

固定四个实际NN路由：raw原forward；feature_ema用同12通道/1/4尺度q=avgpool4(u)、同.5EMA，u'=u+upsample(s−q)；envelope_spatial用当前q的3×3replicate均值替代s、无状态；envelope_temporal如上。feature与temporal首帧须与原forward最大差≤2e−6，另检手动原重组差。空间分支首帧实际处理，不能偷偷设为raw。第五个known_motion_post从缓存raw按已知位移roll前融合RGB，再.5平均，为上下文强参考，非可部署运动估计。所有RGB最终clip[0,1]。

仅用原TrainingPool已冻结57个DAVIS训练视频，按names排序、NumPy RNG SeedSequence([891,0])无替换抽四源；每源抽一个成员和合法128×128裁剪，不读DEV/最终。先固定这四个纹理，所有F0种子与噪声副本共用。每序列T8，GT_t=roll(base,t,axis=x)，中心32:-32评价，边界roll不作自然运动/遮挡代理，GAP仍可能受边界影响。每源photons20/80、两个固定噪声副本，原degrade(gamma2.2×.35、Poisson+σ.008、uint8、无JPEG)，种子SeedSequence([891,source,photons,rep])。另每源一条无噪声且uint8量化的同退化序列。clean只作映射扰动检查，不宣称真实传感器噪声真值。

唯一预算：4源×2photons×2副本=16条noisy，4条clean，各8帧×F0seed0/1×四实际路由=1280NN图；第五post不新增NN。0optimizer/GPU/DEV/最终。CPU2、OpenCV1、FP32、deterministic，全部来自zrh自有环境与已核原F0。运行只允许一次完整固定配置，不以墙钟截止；实现错误保留失败，若修复需单独记录，不能拿结果调参。

记录每序列/每帧中心RGB PSNR；时间指标为t1..7的GT残差按已知+1像素roll对齐后的均方，不采用输出自估field。clean同样记录。主均值为t1..7逐帧PSNR均值和全部差分像素MSE，分F0seed×photons四组，各8序列；clean按seed两组四序列。首帧另存。保存所有路由浮点输出到自有.cache，版本化元数据/标量。

只在以下全部成立时允许另登记训练/DEV：envelope_temporal对raw四组时间MSE比≤.90；对envelope_spatial四组≤.95；对raw、envelope_spatial及feature_ema四组PSNR差均≥−.05dB、各序列最差≥−.5dB；clean对raw两组PSNR差≥−.05dB、最差≥−.5dB。同状态feature_ema时间比和known_motion_post的全部差异完整描述，不把后者当必须战胜的可部署上限。此门槛只是投资筛查，不是独立研究结论。失败停止此固定粗能量/差分/alpha/池化/clip/epsilon的局部网格，不增加loss/训练补救。通过也不能省略新颖性、真实数据、同预算学习、端侧和独立最终验证。
