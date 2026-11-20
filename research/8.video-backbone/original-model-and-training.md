# 原始空间模型、真实继承轨迹与新骨干比较边界

2026-11-20；Codex  (sqs_A100)。依据现有源码、R41/R44实际配置与公开原文；没有打开新数据或最终集。

## 原模型究竟做了什么

`MobileIE/model/lle.py` 的 Slim 路径为：原生分辨率 Conv5(3→12)、PReLU、Conv3(12→12)、FSTS，随后 Conv3(12→12)、FSTS；GAP/Conv1/sigmoid 生成通道门，通道最大值/Conv1/sigmoid 生成空间门，最后 Conv3(12→3)。空间卷积全部 stride1，无通常编码器的步长降采样；因此不能将原模型闪烁直接归因为降采样混叠。边界 padding、GAP 与独立观测噪声仍须区分。

`MobileIE/model/utils.py` 的 FSTS 是两个标量缩放乘积乘以特征平方、再加通道偏置，即 `a*z²+b`；它不自带时间、运动或噪声方差估计。H/W 上的统计在 `MobileIE/loss.py` 中作用于每图每通道的**有符号预测误差**：std/√2 与 mean 决定 detached tanh 权重。它不是逐像素的校准传感器方差。

按实际可训练张量，Slim 共4075参数：912+12+1308+14+1308+14+156+24+327。原论文表列4.047K，差28恰好等于两个FSTS各14参数；这里只报告算术关系，不断言作者计数工具遗漏的原因。卷积预算为 `3828*H*W+144` MAC，1080p约7.938 GMAC；不含激活/门、内存与转换，不能将4075参数解释为几乎没有计算成本。

公开原文表1将 GPU .895ms、SoC 6.72ms 分列，1120.584 FPS 与GPU列的倒数量级相近（并非数值严格互为倒数）；FPS列注明600×400。不能将1100FPS写作本项目手机视频速度；SoC完整输入/精度/delegate仍需原测试细节或实际硬件记录。[原文](https://arxiv.org/html/2507.01838v1)。本文关于该原文的概括仅限上述架构/表项，训练细节以下以本地程序为依据。

## 训练程序与本项目强F0的区别

`MobileIE/main.py` 有10 epoch低学习率warm reconstruction，随后Adam与CosineAnnealingWarmRestarts、验证PSNR择优；`MobileIE/loss.py` 的主loss组合outlier-aware、cosine与PSNRLoss。当前 `config/lle.yaml` 已配置为slim推理，不能直接作为某次官方训练运行日志。默认 `lle.py` 导入 `.utils`，不是另存的 `utils_IWO.py`；后者确有冻结weight加learnable增量实现，但未见默认构造链引用它。因此不能将本地默认运行称为完整复现论文IWO，也不能据此推断作者从未使用IWO。

当前R44 square_mse的真实继承关系：

1. 官方 `MobileIE/pretrain/lolv1_best_slim.pkl`，SHA256 `d40a930fbc0435f47ce08dda17b1c630b86a501be9423b8f48dab14722fb5548`。其LoL预训练已经发生，项目内步数不包含此成本。
2. R41 stage002 的 `training_form_r41.create_model(seed)` **直接从官方slim**恢复到MBR；49978总参数/47239活动参数，warm tail冻结，原函数作为起点。fresh Adam1e-4，MixedPool step0..3999，4000更新。R40配置是精度约定引用，**不是这条权重链的4000更新祖先**。
3. R44从R41完整expanded训练权重继承，fresh Adam且不继承旧moments，MixedPool step4000..7999再4000更新。当前F0是预先指定的square_mse唯一最终，导出4075参数；没有额外时间loss，也不是官方完整warm/IWO/loss训练复现。

所以F0是**官方预训练＋项目内8000更新**，不是随机初始化8K，也不是将R40累加成12K。新架构的教师暖启动若使用F0，须报告教师身份、前向次数和继承的信息；同update数不意味着相同训练FLOPs或官方预训练预算。

## 对主题8的约束

新骨干必须有独立训练空间控制及其自己的half-FB/DIS后处理，不能只和旧F0 raw比；若参数增大但MAC减少，二者同时披露。已知重排、门控、KD和递归均不是独立创新。先用完整因果输出和部署预算判定是否值得继续，再讨论经过外部/实际硬件验证的贡献。原所有局部STOP继续生效，最终封存不变。

本地依据：`MobileIE/model/lle.py`、`model/utils.py`、`model/utils_IWO.py`、`loss.py`、`main.py`；`research/4.temporal-consistency/scripts/training_form_r41.py`、`train_training_form_r41.py`、`bounded_video_r44.py`、`train_bounded_video_r44.py` 及对应stage002/stage001配置。只核训练来源与程序，不重做bit级回放。
