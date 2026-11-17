# R87：保持RGB函数的平方表征缩放自由度检验

2026-11-17；Codex  (sqs_A100)。先登记再代码/图像/优化。[note70](../../5.prior-art-survey/docs/70-incremental-inference-and-functional-gauge-boundaries.md)已限定增量执行、跨层均衡和加权余弦先例；不称本缩放/校准是创新。不是R84 hard匹配/FB/视频融合的重训，不换旧R74 teacher/字段目标，不解封任何旧STOP。本轮只检验一个不同的可学习信息路径：原RGB映射在实数运算中保持，内部用于匹配的度量可以改变。

## 单一结构与训练量

slim F0 head平方前z，h=αz²+b。正对角D缩放head最后Conv3的输出通道weight/bias，head FSTS bias乘D²、body首Conv3输入通道weight除D²，body bias和其余层不变。则原body卷积及后续RGB在实数运算中相同。仅适用于实际12通道slim结构，不能据此声称expanded BN/量化训练等价或任意浮点输入逐bit一致。导出仍4075参数，无新CNN层；匹配/缓存及其它推理成本尚未建立，不称视频零开销。

每个既有R44 F0 seed独立校准12维v，从0开始；a=exp(.5*(tanh(v)-mean(tanh(v))))，故log a均值0、a处于[e^-1,e]范围，匹配有效自由度最多11。描述符为head平方前z的4×4均值池化再每图每通道减空间均值，匹配为Dz的通道余弦（逐向量norm clamp min1e-6）、温度.1。此等价正对角加权余弦，不增加特征信息/函数族，不能把folding本身当创新。

每seed一次全批Adam512步、lr.01/betas(.9,.999)/eps1e-8/wd0/clip1，唯一参数v、F0无grad且始终eval、全固定查询的位移分类CE；共1024次**度量参数更新**，0骨干更新，不写成0优化。只用最终512，不挑checkpoint/早停。日志0/64/128/256/512；不扩矩阵维度/scale bound/温度/噪声/长度/seed网格。

## 固定资料、已知几何与预算

复用MixedPool四个已见训练窗(data seed0/1各step8000/8001)，每窗只取b0/b2的GT第0帧，共8个128×128干净源图块。data seed0两窗的四图为calibration，data seed1两窗的四图为check；check不参加优化/选参。两部分可能有训练视频身份重叠，逐项记录，不称独立未见视频。SYN-source/OLiVES-GT-source只是原图来源；下面两类均重新合成噪声，不是实拍噪声/自然动态验证。loader仍预载完整训练cache/构造原batch，不称只解码256图。

每源图四个quarter已知位移(dx,dy)=(-2,-1),(-2,1),(2,-1),(2,1)，native为4倍。current clean由原warp_previous重采样，部署未知真运动不在本轮建模。previous/current分别用原degrade(photons40,jpeg=False)、SeedSequence[87,图index,位移index,0或1]独立噪声；两F0共享同输入。每对当前描述符固定query坐标{8,12,16,20,24}²，共25；previous候选为quarter半径4的81行优先位移，正类位置来自已知几何。所有查询及候选在图内，16native以上边界使本已知平移内部区域避开原head局部padding；不用GT流估计、难例筛除或ROI调节。

8源×4对×2张=64图/F0；两个F0初次各16次B4=128图。最终D fold后全部同64图/F0再各16次B4=128图，总64 B4/256实际骨干图。每seed calibration/check各400查询；记录baseline/final/fold-check全部检索行，不把同图25点当独立样本。fold-check只是同最终度量的数值资格，不是第三个模型或额外视频质量实验。无新DEV/最终/光流/FB/视频RGB融合/JOD/GPU。

## 资格、必要信号和止损

最终折叠对全部登记输入：unclipped RGB max abs≤1e-4，clipped≤1e-5；canonical余弦(folded descriptors)对显式加权余弦(original descriptors)的logit max abs≤1e-4，所有权重/梯度/输出有限，导出4075参数，F0实际0参数梯度。资格失败保留并停，不换精度/容差/scale bound重跑。指标不是CPU逐bit复现或所有输入的经验穷尽验证。

匹配信号只看check的两来源×两F0四组，固定各200查询：final top1比baseline至少+.05绝对，且CE≤baseline*.95，四组都过才支持另立不同核心设计；不据calibration好看而通过。仅group top1/CE及每query原预测/真类记录，tie按行优先argmax，不另搜统计量。即使过仍不是独立创新/真实flow/视频主表/新正式训练资格，不允许把D塞回已STOP的R84 pipeline追分。任一必要组失败，停止本正对角保持函数度量校准路线的投影/尺度/温度/噪声/长度/seed局部网格，不据此否定所有可学习表征。

CPU FP32/Torch2/OpenCV1、CUDA空mask、确定性，无墙钟截止或科学自动重试；代码/协议执行前冻结，原正常或失败/配置/校准更新/完整查询/权重标记/终态/解释/过程/报告分别文件级提交，阶段推送。变换权重与可重建大资产只在自有data；main/指定作者/no-ff，原R86两经典trade-off/全部旧STOP/保留最终/paper020与完整论文目标active保持。
