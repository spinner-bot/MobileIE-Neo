# 固定权重BN统计重算是已知控制，不是新的视频方法

核查：2026-11-12T18:43:26.795951+08:00；作者：Codex  (sqs_A100)。R48完整止损之后，仅解释强空间基线的大seed/场景偏差，不重开已冻结时间算子网格。

[PyTorch2.9 BatchNorm2d](https://docs.pytorch.org/docs/2.9/generated/torch.nn.BatchNorm2d.html)规定eval使用running统计，默认momentum=.1。[官方SWA说明](https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/)说明权重平均后需用训练资料更新BN。本项目不进行SWA或权重平均，只借用该已知统计工具；不能将其命名为新的时间归一化/新视频贡献。

已阅读[版本2.9.1 update_bn源码](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/optim/swa_utils.py)与本机自有Torch2.9.1+cu126的inspect源码：reset_running_stats，train，momentum=None，遍历输入无梯度前向，然后恢复momentum和整体training状态。累积的是等尺寸批次的均值及各批次无偏方差的平均，不包含跨批次均值差项，因此不能称精确全数据联合方差。未调用的warm BN也重置但不参与部署。

本地检查原MobileIE原MBR包含BN，普通forward不用DropBlock/尾warm头；四帧调用四次B4 BN，momentum=.1时最近窗口累计系数1−.9^4=.3439。这只是运行语义推导，不证明R48 seed差异由BN造成。原模型权重本身也随seed/训练发生变化。R42的GAP/TLC控制不同于BN running-stat重算，note32禁止的新视频BN包装网格仍不做。

R50只固定六个完整训练态、相同训练窗口重算一次，不使用DEV/TEST/手机适配或GT校正、不优化任何参数、不增加部署参数。若指标未支持就停止本校准策略，不通过校准样本量、momentum、域配比或追加seed搜索追分。即使正向也仅是强参考完善；创新、外部视频/夜间、手机与最终验证缺口仍在。
