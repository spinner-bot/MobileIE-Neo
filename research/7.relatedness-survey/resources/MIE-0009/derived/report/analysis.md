# MIE-0009｜reef-track：MobileIE逐帧视频应用的直接工程线索

Codex  (sqs_A100) <langxibielangle@qq.com>；2026-11-20。公开项目适用核心静态分析，0模型/测试/视频/训练/GPU/Gate3运行。

**该项目确实写出了MobileIE slim的前向算子，并接入视频GUI和可选SAM输入；它没有新增增强器时间机制，也没有提供可核验的时序质量结论。** 当前固定包缺少声称已转换的UIEB NPZ权重，因此只能确认结构/调用路线，不能把它当完成数值对账的MobileIE视频部署。

固定[tediyankov/reef-track@af76a927](https://github.com/tediyankov/reef-track/tree/af76a927c568795e800f8d8e0c28c3b939fd6f1f)，元数据2026-11-10，pyproject0.1.0/Teddy Yankov。它是水下鱼类跟踪/礁体重建和标注工程，非MobileIE作者项目。[来源](../../sources/manifest.json)、[阅读过程](../docs/reading-notes.md)、[结构与缺权重核验](../docs/model-and-weight-audit.json)、[主要证据](claims.json)分开保存。

## 可以确认的关联

与[MIE-0002官方RGB slim代码](../../../MIE-0002/derived/docs/upstream-code/model--lle.py.txt)对照，head/body卷积、PReLU、两次FST、全局均值通道门控、通道max空间门控和tail顺序一致；输入RGB/255、输出clamp/uint8。FST非线性保留，没有仅借用名字或用传统LUT冒充网络。该结论限算子拓扑，未做模型数值/设备一致性试验。

实例保存权重/device，既无历史图像/状态也无时间loss或新训练。SAM视频记忆、人工点击后反向backfill与ROI中心平滑均另属跟踪/显示，不能当MobileIE时间恢复。已有逐帧使用不是“时间一致性已解决”，也说明单纯把MobileIE封装到视频不构成二期新方法。

## 需要保留的缺口和条件

|项|固定资料事实|边界|
|---|---|---|
|转换权重|87文件包无mobileie_uieb.npz，gitignore排除npz；固定路径404，当前releases空|文档声称UIEB转换件在目录，不等于已取得；原checkpoint/键/值/转换过程未知|
|4075参数|原模型已知计数、本项目专用测试有该断言|本次未读取缺失NPZ，不冒称对本转换件实物计数|
|可用/测试|available同时要求Torch与权重，GUI禁用不可用项，测试按可用性过滤/skip|总体测试成功也可能没覆盖MobileIE；本调研未运行pytest|
|显示输入|full增强视图宽>960先缩后增强；zoom先native crop增强再显示缩放|不是统一1080p/4K网络成本；SAM独立输入又可能是native crop/full frame|
|时间/ROI|每帧ROI可能重心随动，显示中心/模板有平滑，SAM默认支持backfill|不能把完整交互流程称纯在线因果增强或用ROI稳定代替像素稳定|
|性能|README增强约2–25ms/目标30fps；GUI计时器请求相应节奏|没有MobileIE完整日志/热身/设备协议；M3Pro表的SAM/Wat3R数字不是其时延|

Retinex/CLAHE同样是逐帧对照，其中Retinex按当前crop估计光照/percentile；MobileIE全局均值门控也受crop内容影响。移动ROI/预缩放/压缩可能混入增强观感，实际是否闪烁、多少和原网络内因尚未测量。

FrameMap/proxy记录source_frame=start+stride*i，119.88fps例stride4给29.97fps，而非精确30；它可追溯帧号，时间仍按标称FPS重建并重新编码CFR。原真实视频/GT/增强输出不在包内，未取得VFR逐帧时间或独立画质证据。项目“label-free”也含人点击/修正，输出鱼mask/3D轨迹不是增强GT。

## 对二期研究的使用方式

将其归为**直接算子移植与逐帧视频应用**，和仅有MobileIE启发/通用BYOM标签的工程区分。可迁移的是模型身份/源帧编号/裁剪与下游输入追踪要求；不可迁移的是未得权重的质量/速度、无设备全链证据的4K低开销或新的时序算法主张。组件Apache2.0文本已得，不据此推定整项目/外来权重/私有视频授权完整。

后续可证伪问题：拿到并核定原/转换权重后，固定同视频/时间轴/精度，比较原作一帧函数与移植；再分全帧、固定ROI、随动ROI评估逐帧MobileIE与强时间控制，分别报告空间/时序/下游分割/显示成本。控制人工未来信息、codec、缩放、权重与数据，计完整解码/传输/模型/绘制/内存，不以GUI目标节奏当实测。均为后续实验分支候选，本阶段不执行。
