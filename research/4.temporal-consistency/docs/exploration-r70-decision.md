# R70｜固定传统前处理参考的完整取舍

记录：2026-11-13T06:34:34.052689+08:00；作者：Codex  (sqs_A100)。

原20滤波/896滤波输入帧、40增强CNN序列/1792中心、80质量/3584逐帧、80感知/3584比较帧、全20预定图及原协调器均已正常结束。CPU06:23:30正常3733.936905秒；感知06:27:22正常221.505996秒、GPU5释放；图06:27:53正常257.068083秒，全部20实际查看；协调器06:27:53正常3997.498758秒。没有新SGD/手机或独立最终像素。

表内相对同seed、同输出方式的原R44未前置过滤MobileIE；Etemp比<1表示公共GT变化残差代理降低，JOD为固定原R66显示/名义30fps的预测，非人类MOS。

|域|seed|前处理|输出|ΔPSNR(dB)|Etemp比|ΔJOD|
|---|---:|---|---|---:|---:|---:|
|davis_train_pilot|0|spatial_only|raw|-2.123767|0.805956|-0.123759|
|davis_train_pilot|0|spatial_only|recursive_self05|-2.521613|0.881775|-0.492057|
|davis_train_pilot|0|video_default|raw|-2.496776|0.747820|-0.259653|
|davis_train_pilot|0|video_default|recursive_self05|-2.921914|0.871874|-0.674849|
|davis_train_pilot|1|spatial_only|raw|-2.251927|0.801911|-0.151136|
|davis_train_pilot|1|spatial_only|recursive_self05|-2.570002|0.883152|-0.485925|
|davis_train_pilot|1|video_default|raw|-2.648952|0.745695|-0.294239|
|davis_train_pilot|1|video_default|recursive_self05|-2.975626|0.875818|-0.673767|
|paired_real_ND_DEV|0|spatial_only|raw|-0.563777|0.945293|-0.030946|
|paired_real_ND_DEV|0|spatial_only|recursive_self05|-0.567277|0.974973|-0.045177|
|paired_real_ND_DEV|0|video_default|raw|-0.647401|0.929490|-0.034339|
|paired_real_ND_DEV|0|video_default|recursive_self05|-0.650981|0.968053|-0.049510|
|paired_real_ND_DEV|1|spatial_only|raw|-0.524790|0.944969|-0.021939|
|paired_real_ND_DEV|1|spatial_only|recursive_self05|-0.512901|0.976374|-0.037081|
|paired_real_ND_DEV|1|video_default|raw|-0.610044|0.930335|-0.024711|
|paired_real_ND_DEV|1|video_default|recursive_self05|-0.595560|0.970299|-0.041316|

两mode/两域/两seed/raw与rec全部16个均值条件的PSNR/JOD变化均负，时间代理降低。SYN raw少2.124–2.649dB、JOD少.124–.294；rec少2.522–2.976dB、JOD少.486–.675。ND少.513–.651dB，JOD变化约−.022..−.050；这些小感知均值不证明人类显著差异。没有稳健质量—时序—感知联合提升，也不能把降低时间代理自动当作成功。

所有20静态图的全FOV/原生中心均已实际查看：SYN彩色颗粒减少，毛发/石纹/草地/人脸和建筑细节也更平滑；walking frame32的过滤后rec建筑边缘出现可见变形纹理。ND差异较小、部分色调/细节问题仍在。静态图不证明真实闪烁或具体伪影因果，未人工完整播放/MOS。

保留原核/固定盲sigma两个mode的完整传统参考，按事前协议不扩sigma、窗口、滤波强度、裁块、权重、长度或seed网格，不选有利序列重报。不替换原MSE+经典rec强参考，不回写任何旧候选门槛；新方法仍未确立。

本轮并非最优BM3D/VBM3D复现。spatial_only只是同VBM实现关闭时间搜索；video_default是原默认Nf4的整片离线任务适配，Nf不是经证明的端到端未来等待界。首帧Haar-MAD标量并非已知AWGN/真实ISP标定，合成Poisson/clip/量化和实拍相关噪声不满足简单假设；固定增强器还可能受到前处理分布变化影响。没有隔离这些原因，不能将结果推广为传统去噪无效、去噪—增强分解错误或某个单因素根因。

描述性CPU2全过滤阶段中位秒/帧：SYN480×854 spatial1.823989/video5.249411；ND288×512 spatial.676927/video1.774088。同一过滤结果跨两个seed复用，过滤只计一次，不能把缓存复用当每帧免费。见classical-cost完整12组/各stage min/median/max。共享CPU/数组IO及未独立预热重复的限制保留；不是手机时延、实时能力、完整FLOP或1080p/4K外推。

本轮完整表300质量条件/60均值、280质量条目匹配JOD，448唯一观察不变；旧FastDVD five_current缺JOD不补值，R69另40行不被默认为本表已含。全部输出/来源/接受域与成本保留。R72原固定REDS参考继续，不因本结果改其数据/权重/预算。待完整跨来源证据后再选择有界方法假设；不以此结束创新、自然动态/手机和独立最终目标。

代码/完整结果/图/actual review/决定/过程/进展报告分别密集提交，阶段fetch/SSH推送、显式no-ff/指定作者/自有zrh data保持，不改移动演示。
