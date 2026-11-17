# 69｜快速传统运动参考：不能用Farneback成本概括全部flow

2026-11-17；Codex  (sqs_A100)。R85只测现有原half-FB(Farneback)参考，并未比较所有传统运动算法。局部rg搜索research4/5及specs未发现DISOpticalFlow/Dense Inverse Search已有版本化实现或报告；该检索不是全互联网穷尽证明。

[DIS原作者arXiv1603.03590v1摘要](https://arxiv.org/abs/1603.03590)提出逆向patch搜索、稠密聚合和变分细化的经典方法。本次只读正式摘要，没有读PDF或复现原文速度，不采用其Hz数字来替代本服务器或手机测量。

[OpenCV4.11正式API](https://docs.opencv.org/4.11.0/de/d4f/classcv_1_1DISOpticalFlow.html)实际核对预设、pyramid finest scale、patch配置、均值归一化、空间传播和初始flow接口。文档区分三个预设，且说明实现包含对原算法的扩展；不能称直接复现作者实现。创建本项目现有OpenCV4.11 ULTRAFAST对象、仅读取getter，实得finest_scale2/gradient_iterations12/patch8/stride4/mean_normalization=True/spatial_propagation=True/variational_iterations0。未调用calc、传入图像或新增CNN，没有安装/下载任何外部依赖。

这是一个明确传统参考缺口：下一可登记一个固定DIS ULTRAFAST而非preset/尺度网格，保持原F0和因果RGB融合，并同时审查画质、时间、接纳与完整计算成本。若仅更快而画质/时序明显损失，就如实保留trade-off，不把已知DIS包装成新方法，也不据它解封R74/R84失败配置。R85所见1.79–1.91倍完整CPU成本只属于当前half-FB参考，尚未证明传统运动普遍不可部署；真实手机与独立自然动态结论仍缺。
