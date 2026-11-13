# R47 公布网络的CNN预算口径

登记：2026-11-12T18:10:06.938236+08:00；作者：Codex  (sqs_A100)。R47/R48尚在训练/最终评价，仅补论文参数/MACs列，不改活跃代码或训练预算。

固定batch1输出帧，在已有SYN480×854、ND288×512、phone360×640和标准1080×1920四个分辨率作CPU Torch meta形状前向；零像素/零质量/CUDA不初始化。MobileIELLENetS原4075参数与固定FastDVDnet源码2479096参数，按官方反射补齐4倍尺寸后推理形状。只对Conv2d按output_numel×(Cin/groups)×kernel面积计MAC，一乘加记1MAC；排除bias加法、BN/ReLU/注意力/池化/条件map/flow/warp/IO，不能叫完整pipeline FLOPs、实测时间或手机性能。

FastDVDnet同一个temp1 DenBlock每输出执行3次，temp2执行1次，应按调用计算而非唯一模块只数一次。额外给出由已登记window_indices计算的16/64帧上下文中可复用第一阶段三帧tuple的理论算术次数：它不是现有适配器已实现的缓存、计时或峰值内存，不用其推断速度。five_current的重复tuple也有潜在复用，信息消融和运行成本要分开。

所有张量大小仅是单个conv输出的FP32体积或输入窗口逻辑元素，不是实测峰值、完整storage或持久历史。报告既有MobileIE1080p计数应复现7937740944 MAC；公开参数/源摘要与原调用结构核对。论文只在完整质量后合并相应表，不用meta计算替代研究贡献。
