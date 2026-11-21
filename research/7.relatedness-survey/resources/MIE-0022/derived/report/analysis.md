# MIE-0022｜JND预滤波：极小MobileIE骨干与感知训练的直接先例

**主体预印本和可得适用实现分析完成，建议证据受限结项，待Gate2另轮审查。** 完整v3五页、4图6表和固定作者工件已分析；作者逐视频补充CSV的分享可访问，实际归档内容未取得。0研究源码执行/模型加载/训练/GPU/Gate3。

[完整阅读](../docs/reading-notes.md)、[版本与作者名单](../docs/version-review.json)、[实际图表审阅](../docs/visual-review.json)、[源码阅读范围](../docs/source-reading-scope.json)、[原始工件](../docs/author-source-manifest.json)、[参数及损失口径](../docs/static-arithmetic-and-loss-review.json)、[数据元信息](../docs/dataset-metadata-review.json)、[补充资料获取](../docs/supplement-acquisition-review.json)、[主要主张](claims.json)。

该文p3明确采用MobileIE[33]骨干，改小卷积核和通道，并用传统JND teacher伪目标及参考独立的频域单边幅度约束训练。它构成“轻量MobileIE+感知/频域训练”的直接先例，简单组合不足以作为二期创新。约束不涉及系数符号/相位或连续时间，也不是人眼闪烁阈值验证；当前实验只滤Y、All-Intra、默认首帧，TM类是ones占位。

当前Lite字面参数24949训练/1861折叠与表值吻合，但baseline当前5×5首层为41413/3257，论文为38.92K/3.06K；全3×3假想版本的算术吻合只是解锁线索。发布权重首卷积操作码也显示baseline5×5/Lite3×3。源码conservation loss为线性hinge而论文Eq6是平方，MS-SSIM的pred单边softclip未在论文交代；默认K10掩码虽非交替zigzag遍历，集合仍一致，不能误判。

码率结论有明确口径：表III六指标ALL、表IV/V四指标ALL不同；-2.80与-2.47可由指标集合解释，不是互相矛盾的同口径结果。感知码率平均改善与部分PSNR码率增加并存，目标函数与时序稳定性不能互代。当前更大骨干的公开结果对应BaseLine+Lf，不是仅空间loss的BaseLine。

7.15GFLOPs和RTX2080Ti的10.88ms是作者桌面网络数据，14.1%相对IQNet50.70；不是手机或完整编码链成本。单文件入口每帧加载模型，批入口每video复用；实际pad/归一化/round/读写/codec/指标/BD聚合都需要计入未来协议。原图和teacher、训练验证best、原始曲线与表内均值分开。

主体v3已全读，v1/v2原件与差异保留；当前仅Submitted to ISCAS2027，未确认正式录用。150作者工件全部blob匹配、46适用全文及明确的部分路径静态读，3PTH仅CRC/操作码。作者逐视频results.tar.gz分享显示1070637B，公开码验证成功、普通下载确已点击，仍无实际文件；独立替代检索未定位另一合法载体，缺口保留。不能称补充不存在、已读或原始BD数据已独立复算。

未来控制需先解决公开baseline/训练/导出版本，再分离平方/线性约束和等权MSE，固定数据/teacher/时间信息/质量/完整预算，考察相位/曝光/真实运动噪声及时空联合指标、手机热功耗和完整编码链。当前只交接可证伪问题，不提前全资源综合。
