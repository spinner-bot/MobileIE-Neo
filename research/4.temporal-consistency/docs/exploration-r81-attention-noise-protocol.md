# R81训练来源静态噪声干预：门控是否介导足够输出波动

2026-11-17；Codex  (sqs_A100)。本登记在读取本轮训练图块/生成噪声/模型前提交。R80完整失败及全部旧STOP保持；本轮不是新部署方法、正式训练、DEV/最终评价或局部门控参数网格。

## 一个窄问题及数学前提

MobileIE的通道max满足 `|max(a)-max(b)| <= max|a-b|`；argmax索引改变不意味着max数值或网络输出不连续。原att1是1到12通道的逐像素Conv1+Sigmoid，不能凭索引切换就认定其为闪烁根因。R24只固定当前全局gate于另一个聚合网络，R80固定当前gate拟合13全局系数；均未直接隔离当前空间gate对输入独立噪声的介导作用。

本轮在完美静止、已知同场景无噪暗输入条件下直接干预节点：保持每次noisy head/body特征不变，仅替换全局gate或局部max的值，观察输出随机波动和重建误差。替换参照需要clean训练图，故不可部署；它不保证数学最优、不给出所有attention模型的上界，也不研究实际运动/真实传感器噪声。

## 固定输入、模型与预算

沿用R32正式训练清单的57个DAVIS源，按名称排序前两个源的第0帧、中心128裁块，不按图像/成绩选择。只解码这两个已有训练RGB图像；不读取DEV/保留源或另取像素。两源各photons20/80，用原degrade、gamma2.2/乘.35、Gaussian.008/clip/uint8、jpeg=False，每条件32个独立draw，SeedSequence([81,source_ordinal,photons,draw])。噪声前参照同暗化/clip/uint8但不加Poisson/Gaussian；它不等于clipped随机观测的严格期望。

两个R44 square_mse最终4075参数/累计8000权重，全部eval/冻结。两个模型使用相同两训练图块和噪声数组，共8条件。每条件32共享noisy core，共256；两模型×两源各一次clean core共4。每clean及每个条件第一个noisy另执行原完整forward作函数比较，共12额外原forward；合272实际骨干调用。每noisy core的raw及三干预各一次tail，另4clean及12原whole tail，总1040 tail调用。0SGD/拟合/新GPU，全CPU2/OpenCV1、初始CUDA mask为空、确定性FP32。

令 `u=body(head(x))`，`g=att(u)`，`m=max(g*u)`；无噪暗输入给g*和m*。五个输出：

1. raw：`tail((att1(m)*g)*u)`，保留原Slim乘法次序。
2. clean_global：把g替换为g*，从g*乘当前u重新计算m，再原att1/tail；局部值允许随其上游改变。
3. clean_local：把m替换为m*，当前g/u不改。
4. clean_both：g*与m*同时替换，当前u不改。
5. recursive05：原raw先clamp[0,1]，首帧raw，后续逐帧`.5*current+.5*previous_fused`；完美静止对应，不跑flow、无未来。不是实际动态强post的原文/手机复现。

原节点重组对原forward最大归一化差≤2e-5、所有参数不变/有限，raw或干预非有限则保留并停止，不调整容差重做。每组输出clamp[0,1]后作同GT RGB重建；连续32draw按固定顺序计算平均相邻RGB绝对差，GT静止故无GT运动项。保留均值PSNR、MSE、时间差、总体逐像素方差/均值偏差分解（用于解释，不以逐bit精度作研究成果）；四前向路径另记录unclipped方差，排除只看clip造成的表面降噪。记录winner相对clean参照变化比例及max值变化大小，但不以相关性认定根因。

40条件输出行、1280逐draw输出帧比较，全8条件/两seed/两源/两噪声保留。运行数组放自有cache，Git保留来源/参数/40指标/必要逐draw数值与终态；八张固定noise0/1/31静态对照必须实际查看，非全视频/MOS。256噪声样本按生成数组独立计数，不称256个真实视频帧。

## 先验判断规则与停止范围

只有clean_local在全部8条件相对raw时间差比≤.90、平均PSNR差≥−.05dB，才有继续论证具体空间门控时间路线的最低信号；recursive05及global/both作为完整解释控制，任何有利项都不能替代主条件。通过也须另提出实际可获得的因果状态、运动/成本和先例差异，当前没有自动GPU/质量训练许可或新颖性结论。

若未通过，降低只平滑原局部门控的优先级，不追加max/均值/softmax、温度、滞回阈值、EMA、种子、draw/源数量等局部网格，不拿更多CPU诊断代替方法收敛。此为固定无噪节点干预的有限结果，不是否定重训/共同变化模型的普遍定理。任何后续不同核心机制须有独立理由；原R64/R73/R79/R80停止、最终像素/自有data/main/密集分别提交/no-ff与阶段推送持续有效。
