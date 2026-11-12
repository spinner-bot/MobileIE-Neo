# R65｜已发表形变一致性训练的共同任务适配

登记：2026-11-13T02:49:45.665964+08:00；作者：Codex  (sqs_A100)。登记时尚无R65代码执行、训练更新或评价。R64已经完整止损，不重开其状态/宽度/目标网格。R62原clean权重只有零更新迁移结果；本轮补必要的已发表增强器训练参考，不把适配当新方法或论文目标完成。

固定StableLLVE commit `4cc070da328ffadc8498285985aa98175e473bc5` 的原4316259参数U-Net与既有公开clean权重（R15 manifest）；两个arm、两个seed均重新从同一公开权重开始，全部参数开放，无BN/新模块/历史状态。保留BGR、FP32，训练和评价不作GT亮度或颜色修正。公开预训练包含开发来源重叠的既有风险，不能声称完全未见泛化。

## 目标和形变

`spatial_l1`：四帧原始未clip输出对GT的平均L1。`commute20`：同一空间L1，加20倍平均 `|F(W(x))-W(F(x))|`；保留两条预测路径梯度，不detach、不对形变图另加监督、不clip训练输出。权重20和L1来自官方train.py，非本项目选择出的最佳值。

原MixedPool四帧每个B4分别调用网络。形变只使用前三源帧：对每个相邻训练GT估计half_forward_backward的反向flow，用它将源帧t重采样到下一坐标；它仅作为变形场，不用下一输入代替变形图。每次一致性平均B×3×RGB×H×W。第四原帧仍参加空间L1。没有利用真实跨帧独立噪声的额外监督；此限制须解释。

保留官方warp.py反射边界和endpoint linspace网格，明确align_corners=False（其PyTorch1.5默认），不掩膜/裁边；不能默默换成项目transport的边界处理。该旧坐标约定零flow也不严格为identity，CPU资格记录这一差异而不改几何救分。训练flow取已裁块/翻转后的GT，未使用DEV flow、CMP模型、instance mask或新公开资产；flow精度不是本轮比较对象。

因此本轮是**官方网络及损失形式的固定预算任务适配**：不同于原论文单图CMP形变、512²裁块、随机初始化与50epochs；不能称原论文充分复现/收敛。对MobileIE还存在参数规模、L1/MSE、预训练、训练参数化和计算量差异，不能把交叉排名解释为单独容量效应。只有本轮两arm内部提供损失效应控制。

## 固定资料、预算和数值

严格原MixedPool.batch(seed,data_step)0..7999、B4T4crop128、50/50 SYN/real：前2来自既有57DAVIS训练源，后2来自12ND训练源，排除bear/lucia/walking及15/47/91/121开发视频；不改变原噪声、量化、裁块和翻转。每seed两arm各8000正式更新，共32000；每步两arm复用同一输入/GT/flow。Adam1e-4、betas(.9,.999)、eps1e-8、wd0、global clip1；4000后重建Adam，与既有8000控制的优化阶段一致。FP32/TF32off/确定性/CPU2/OpenCV1。不做长度、LR、权重、数据比例、种子或形变网格，只评价最终8000；每1000为备份。

CPU先在seed0/data_step0执行两arm各1次丢弃SGD，验证原权重输出/形状/有限性、重建和一致性两路径梯度、首尾层梯度/实际参数变化及官方warp几何；这些更新不继承。每个GPU作业在CPU资料及首窗准备后作内部新双采样，随后两arm各1次丢弃CUDA gate，再重新载入公开权重和Adam。因此总正式32000与6次丢弃更新分开。资格失败保留，不放宽目标或阈值追分。

共享GPU只用同组4–7，最多四张，预期两张各一个seed；外部新guard双采样（无compute进程/≤64MiB/≤5%）且保持项目锁，内部准入在CPU准备后且CUDA未初始化。无墙钟截断，科学失败不自动重试，不修改活跃依赖。

## 最终评价和边界

原六SYN64帧与四ND16帧，共448观察/arm；每arm保存raw及自身raw half-FB recursive .5，首帧raw、无reset/GT修正/未来等待。共同CPU2/OpenCV1评价，两seed×两arm共40CNN序列/1792中心、80质量行/3584逐帧行。沿原GT流/接受域与PSNR、SSIM、GT变化残差Etemp、edge定义，全部源/seed/方法保留。没有新手机/Night/DRV/最终测试或移动端时延；CPU神经计时不替代完整流水和手机成本。

完整后比较commute20与同seed spatial_l1的raw和rec，同时列R62公开零更新、R44/MobileIE、R56/Nano及R53/FastDVD既有条件，披露差异。不预设“适配必胜”或以该参考设新方法采用门槛；无论结果如何，本轮不追加上述局部网格。不以更平滑或warp代理下降替代画质/感知闪烁证据。

协议、代码、资格、原始终态、表/解释、过程和进展报告分别密集提交，阶段fetch核分歧后推送；merge/pull显式--no-ff、指定作者、自有zrh data、project-python入口，不改移动演示。目标仍active，独立方法创新、可信联合收益、标准/自然场景验证与手机测量仍待完成。

来源：[官方train.py](https://github.com/zkawfanx/StableLLVE/blob/4cc070da328ffadc8498285985aa98175e473bc5/train.py)、[dataloader.py](https://github.com/zkawfanx/StableLLVE/blob/4cc070da328ffadc8498285985aa98175e473bc5/dataloader.py)、[warp.py](https://github.com/zkawfanx/StableLLVE/blob/4cc070da328ffadc8498285985aa98175e473bc5/warp.py)、[项目README](https://github.com/zkawfanx/StableLLVE)。
