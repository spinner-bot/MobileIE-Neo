# MIE-0013｜MobileIE继承实现的默认协议与复现条件

[作者固定发布](https://github.com/LethyZhang/FGDPA/tree/530c692feb99ecf6205bb929118a9167e2809d4d)包含完整适用模型/训练/数据/loss代码、slim权重、90项test名单和1000轮日志。它与[MIE-0012论文](../../../MIE-0012/derived/report/analysis.md)属于同一证据族。源码直接继承FST/多分支折叠并增加FFT注意力，是明确关联工程，尚未独立数值或设备复现。

当前RGB→FST/CNN→当前图FFT统计与空间max/mean门控→输出，没有时间状态或时间loss。FFT空间频谱、BN运行统计和模型实例并不提供历史恢复。典型输入池化到32²，但非整除先resize、边小于32会回退；全分辨率attention为C通道，固定FFT尺寸不意味任意输入不变或常数总成本。slim维度字面4234吻合论文，FST/FFT仍在推理，不能说所有运算已折叠。

[条件静态推导](../docs/prior-dct-initialization-review.md)及本项调用核查显示，DCTLinearBranch默认PW和gamma同时零，fresh/有限梯度/无非零加载时两者梯度均零；这限制当前源码对DCT训练消融的解释。不是历史实验错误的证明，其他卷积/attention可训练，且未执行任何反向传播或训练。不得悄悄改初始化后称原版复现。

默认配置是original/pretrainedFalse/need_slimFalse，README测试命令不会自动使用所发slim权重；strictFalse也不是正确加载认证。90名单不由loader自动应用，需事先按目录分割；代码无论文翻转/旋转，当前lr.001与日志.0009不同。日志最佳657约23.969是作者文本证据，缺源码/config/权重历史链；权重只核归档/容器，未加载。demo的三项Dataset返回与两项解包另有静态接口风险，gt=None也进入默认collate；未实际运行定位首个异常。

可迁移公开模块/折叠流程与可追溯划分线索，不能直接迁移论文消融/吞吐、猜测历史state或把其当时序先例验证。后续应先核确切版本、初始化、结构/权重加载与split，再同质量/时间信息/运动曝光条件评价真实video及FFT后端、全链时延/内存/能耗；修正版单列。所有实验尚未执行，本次0源码执行/模型/训练/GPU/Gate3。

[过程](../docs/reading-notes.md)、[21文件相同摘要与范围](../docs/source-correspondence-audit.json)、[原件清点](../docs/archive-inventory.json)和[主要证据](claims.json)保留。Apache2代码文件存在，数据/权重来源授权及完整环境另有缺口；已提供的实物不等于全部科学结果验证通过，亦不因缺环境而称整个实现不可得。尚未进行全资源总体综合。
