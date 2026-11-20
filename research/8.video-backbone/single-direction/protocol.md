# S1：单向field的目标贡献者离散度筛查

2026-11-20；Codex  (sqs_A100)。先登记再实现/冻结/首次执行。[先例与信息问题](../../5.prior-art-survey/docs/74-one-direction-transport-and-ambiguity-priors.md)。不主张前向归一化/flow方差本身创新，不重开旧preset/阈值/alpha/匹配/训练网格。

原R44同seed最终4075参数F0全部冻结；原MixedPool seed0/1、step8000/8001、B4T4/128、每批两SYN两ND，64实际NN图、16窗口。原排除及封存不变，不读新DEV/最终、不训练/GPU。CPU2/OpenCV1/FP32/deterministic，不设墙钟截止。

每窗相邻原enhanced raw经原灰度/round/native DIS ULTRAFAST配置得到back与forward。共48双向对/96calc，完整参考和新路径共享同field值，不能按实现共享的运行时间宣称单向实际更快。GT帧原half_forward_backward仅构造共同评分field/接受域，共48对/96Farneback调用；这不是物理GT flow，也不进入任何候选推理。

五条递归输出，首帧均原raw、融合alpha固定.5、每窗独立自己的previous_fused：raw；original_FB使用原native FB接受和原pull；backward_only只用back和原bounds、取消FB；push_average只用forward、双线性四角scatter至目标，以质量和归一化；push_dispersion与push_average同值，但另累计该目标贡献者的forward位移均值μ和二阶范数均值，v=max(E||f||²−||μ||²,0)，只在质量>1e−6且sqrt(v)<1+.1||μ||时接受。其他像素回退当前。这里的“质量”是splat mass，不是画质。flow单位native像素，不把v当校准运动后验或传感器噪声。四角只保留在目标图内的正权重，FP64累计、归一值再FP32；不clip贡献质量，RGB自然凸融合。

实现资格固定三类纯数组、0NN/flow：identity常量保持；常量(.25,−.5)位移目标方差≤1e−10/常量保持；16×32两半各向中间±4像素的碰撞，重合区离散度须>0并被上述mask拒绝。另将original_FB输出和原RecursivePipeline在完全同已缓存fields驱动下比较，归一最大差≤2e−6；不再重估flow。所有输出finite、shape/首帧恒等、0..255范围。

使用原exploration_r2_data.evaluate和GT-derived共同flow/mask，保存全帧PSNR/SSIM/边缘和残差Etemp；组均值分seed×SYN/ND四组各四窗口。GT mask空支持须显式失败，不默默弃样。逐窗/逐帧数据、接受比例、贡献者离散度分位数、原权重来源和全部浮点预测留存。新候选唯一为push_dispersion，不从其它输出中事后选冠军。

允许进一步完整DEV/成本登记仅当以下全部满足：候选对original_FB四组PSNR差≥−.05dB、最差窗口≥−.5dB、Etemp比≤1.02；对同单向calc数的backward_only四组PSNR差≥−.05dB/最差≥−.5dB且Etemp比≤.95；对push_average四组PSNR差≥−.05dB/最差≥−.5dB且Etemp比≤1.00。没有要求必须全局胜过用真运动的oracle，没有MOS/手机/独立最终结论。若失败，停止本固定目标贡献者方差/质量界/阈值/alpha/preset/field来源局部网格，不训练补救，不扩失败DEV/成本/手机。通过仅允许独立登记下阶段，不能说已建立创新或加速。
