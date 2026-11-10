# 特征乘法、噪声相关性与下一轮结构假设

- 创建：2026-11-10 21:10 +08:00 · Codex  (sqs_A100)
- 触发：有界平方变换的收益接近更强时序正则且增加延迟，转而检验时间信息是否应进入特征变换本身。
- 定位：定向调研、数学动机和待验证结构假设，未实现有效的视频模型，不构成新颖性证明。

## 一手工作核查

- [NAFNet论文](https://arxiv.org/abs/2204.04676)及[作者SimpleGate代码](https://github.com/megvii-research/NAFNet/blob/main/basicsr/models/archs/NAFNet_arch.py)：本轮检查了官方代码中的通道分组乘法及所在block。乘法特征变换本身已有成熟先例，不能仅把FST平方改成双分支乘法就作为创新。
- [TSM，ICCV2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Lin_TSM_Temporal_Shift_Module_for_Efficient_Video_Understanding_ICCV_2019_paper.html)：官方摘要/论文概述有在线因果通道移动设计。其视频理解性能不能直接推导为视频增强有效；应将时间shift作为低成本对照，报告访存和状态成本。
- [Approximated Bilinear Modules，ICCV2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Zhu_Approximated_Bilinear_Modules_for_Temporal_Modeling_ICCV_2019_paper.html)：官方索引摘要已覆盖时序双线性建模；全文页访问403，未精读实现。泛称“时序乘法/双线性”已有相关先例。
- [Noise2Noise](https://arxiv.org/abs/1803.04189)：本轮核查官方摘要，说明噪声观测与统计假设值得单独研究；下述乘积推导是本项目的基础数学核查，不归为该论文提出的方法，也不作为新贡献。

## 从自平方到两个观测的乘积

在简化条件下，令两个观测为 `x1=u+e1`、`x2=u+e2`，噪声零均值、同方差sigma²，相关系数rho。则：

```text
E[x1*x2] - u² = rho*sigma²
```

原自平方相当于rho=1，噪声引入sigma²的偏差；两个独立观测rho=0时，该项消失。

进一步假设联合高斯，则相对u²的MSE为：

```text
2*u²*sigma²*(1+rho) + sigma^4*(1+2*rho²)
```

CPU微验证使用45种固定条件、每种200000次采样，偏差与MSE均通过闭式结果的统计检查。见[原始数据](../../4.temporal-consistency/results/exploration-r2-scalar-moments.json)和[代码](../../4.temporal-consistency/scripts/probe_scalar_moments.py)。这只是标量数学验证，不能当作第45组模型实验或实际去闪烁效果。

## 为何不能直接推出可用的视频结构

1. 神经特征的噪声可能非零均值、非高斯；同一帧两个卷积分支通常也并不独立。
2. JPEG/视频压缩、曝光和传感器处理可能产生相关扰动；历史状态也可能与当前输入相关。
3. 运动、遮挡或光照改变使两个时刻的真实特征不再相等，交叉乘法可能引入新的偏差和拖影。
4. 对齐、状态存储和可靠性判断要计入完整开销，不能以乘法次数相同宣称零推理成本。

## 下一轮可证伪的探索

- 首先在已训练网络的真实中间特征上测均值偏差和跨噪声/跨帧协方差，加入相关噪声和运动，不直接跳到复杂网络。
- 若假设部分成立，再把FST内部的自交互改为受可靠性控制的跨帧交互，作为对原骨干的结构适配候选；使用初始函数匹配及普通训练/时序损失强对照。
- 同时比较同帧双分支、输入拼接、因果shift、普通特征EMA及输出融合，区分交叉统计、时间信息和参数增加。
- 若理想对齐下也无稳定收益，或普通正则解释了收益，则降低该候选优先级；若收益只来自oracle运动，不能作为可部署方法结论。

## 修订记录

- 2026-11-10 21:10 +08:00 · Codex  (sqs_A100)：记录相关工作的直接重叠、基础推导与标量验证，并明确尚缺的神经特征和真实视频证据。
