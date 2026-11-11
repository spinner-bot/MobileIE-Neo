# 25｜骨干计算分配、流式状态与已知空间强控制

记录：2026-11-12 04:42 +08:00；作者：Codex  (sqs_A100)。R28主路线已冻结；本次补读固定源码以约束下一骨干设计，不将已有组件组合写成创新。

## NAFNet与原FST的区别

[NAFNet v4正文](https://arxiv.org/html/2204.04676v4)§3–5及[固定代码](https://github.com/megvii-research/NAFNet/blob/2b4af71ebe098a92a75910c233a3965a3e93ede4/basicsr/models/archs/NAFNet_arch.py)已核查。SimpleGate乘通道两半；NAFBlock已有pointwise/depthwise卷积、LayerNorm、简化通道注意力及零初始化beta/gamma残差缩放。MobileIE的FST是同一特征自乘且共享标量曲率，两者函数类不同；更一般乘积、深度卷积、零扰动初始化都不能单独主张新颖性。

NAFNet实验段明确讨论裁块训练/整图测试的全局统计不一致，并使用TLC。[TLC一手论文入口](https://arxiv.org/abs/2112.04491)提出测试期局部化全局操作。本次对TLC仅核查摘要及NAFNet相关正文，尚未按原代码复现MobileIE转换。MobileIE有GAP、既有训练用小裁块，需考虑该已知空间强控制；但R28失败不能据此归因GAP，旧R24冻结门控干预也不能替代这个控制。

## BSVD源码补读

[BSVD固定版本](https://github.com/ChenyangQiQi/BSVD/tree/29a6f050c14b48ccfe879ec7c7e20fa79e28e910)的`bsvd_arch.py`、`tsm_arch.py`与`profile.py`已只读检查。默认双向类由两个DenBlock组成，每个四个MemCvBlock、每块两个BiBufferConv；按拓扑推导共16个buffer层。streaming_forward追加None并丢弃shift_num个前导输出，不能把此类的流水计算时间当无未来等待。独立unidirectional配置尚未复现，不将默认双向结论套到所有变体。

BiBufferConv同时保留完整center与旧center的通道切片，另有MemSkip队列。依据[PyTorch2.9 view语义](https://docs.pytorch.org/docs/2.9/tensor_view.html)，切片可能保留完整底层storage；不能只按C/8逻辑元素估计缓存。这不构成作者峰值内存测量错误的证明，实际峰值仍需测量。后续自有状态应核对唯一storage、队列和边界flush；该实现还硬编码CUDA，官方profile自动选显存卡，本轮未运行或安装它。

## 本项目预算分析与接续约束

[meta图预算](../../4.temporal-consistency/report/architecture-budget/tables.md)只计算未训练空间控制：原模型4075参数/7.938卷积GMAC@1080p，替换body为深度可分离后3043/5.773，替换head第二层与body后2011/3.608。原两个12→12密集3×3层占约67.7%逐像素卷积MAC，有计算可重新分配，但替换不能精确分解任意预训练卷积，质量与手机速度均未知。

12通道全分辨率FP32历史张量下限94.92MiB；示意8通道四分之一长宽张量FP32/FP16约3.96/1.98MiB，尚非实现的完整视频模型或峰值内存。目标改为匹配总成本的空间/因果骨干控制，保留history替换以检验真实信息利用。固定teacher flow是既有公平比较协议，并非经典post部署必须额外跑一个独立CNN；后续效率参考也应考虑共享自身增强输出估计运动，不能人为抬高基线成本。

57段公共训练资料已登记并选择性获取，现有三开发片排除；本次不启动新模型训练。下一步在这些约束下选少数结构与同预算训练方案，另行登记，不再追加R28相似算子网格。
