# I1-S：已发表Simple learned稳定化的八帧任务适配

2026-11-21T15:19:55.006227+08:00；Codex  (sqs_A100)。先于Simple模型前向/资格优化、T8训练像素加载与GPU。本协议新增作者SimpleLearnedStabilizer及固定配置两个原文件，原controlled接入6源/作者21源不改。[前阶段费用决定](controlled-preflight-decision.md)保持，不缩宽/改尺度救controlled，也不解封R5或任何旧STOP。

## 固定模型、数据与目标

两个作者标准层级：all按原HDRNet simple配置在原MobileIE全部6个Conv2d及root输出各加每通道EMA；output仅root输出同一EMA。原F0 seed0/1分别完整权重冻结、forward保留中间梯度，不学习空间骨干。logits固定初始化4、sigmoid作为当前权重，首帧精确直通；使用原作者模块和reset，不新增flow/controller/未来/门预测。预计全模型4141/4078参数，新增66/3，解析Conv均原7.937741G/1080p，EMA算术/复制/历史状态仍有代价。参数和这些费用不能作手机速度；旧4075非用户硬上限。

新T8采样复用原MixedPool加载器的57个SYN/12个ND训练源和原degrade，原T4函数不改也不调用。原SYN3428帧/min25均已登记，不读DAVIS testdev或封存最终。B1T8/128，step0..79999偶数SYN/奇数ND，域各40000批；同seed两模型读相同批。SYN随机源/连续起点/固定八帧同crop和flip、photon20/80、原gamma2.2×.35/Poisson+Gaussian.008/无JPEG；ND从16帧取连续8的9个合法起点，同crop/flip。固定stateless RNG SeedSequence([194, seed, step])；采样seed即F0 seed0/1。来源和全部窗口身份保留至自有cache，不以64个窗当独立场景。不推断实际传感器fps。

原作者UnifiedLoss/RMSE epsilon1e−6、λ.4，逐帧RMSE均值加相邻预测RMSE的相同1/T归一，不用warp/GT flow/附加MSE/KD。不clip训练raw预测；实际数据已经8bit/截断，范数理想定理不作为本数值优化保证。两模型各fresh Adam lr1e−3、(.9,.999)/eps1e−8、wd0；step40000/60000前降到1e−4/1e−5，无梯度裁剪/权重择优。每seed两模型各80000，共320000正式更新、2560000学生帧；40000个SYN与40000个ND八帧窗/seed。τ8/80K及SL学习率来自作者增强设置，资料/任务/F0与作者HDRNet/NFS不同，不能冒称原成绩复现或与旧4K训练等预算。

CPU资格两层级人工B1T8/24×32、各2共4丢弃更新，核首帧/禁用/reset/未来前缀/重复当前、原权重冻结、梯度更新及真实参数/状态量，不用loss选择层级。先materialize lazy参数再Adam，资格权重和Adam全丢弃。随后一次CUDA资格：seed0真实SYN/ND各一T8批，两模型各2共4丢弃更新；实际PID/UUID匹配、allocator lifetime≤2GiB、自身最终NVML≤4GiB、所有loss/梯度/权重有限；只过后fresh正式。资格若资源暂停接同预算，不把部分资格权重带入正式。

## 执行、恢复与成本边界

CPU2/OpenCV1、FP32、确定性/TF32 off、CuBLAS工作区在进程启动前配置。物理1/2（同0–3组）最多两个数值worker；父进程绑定UUID、11×2秒准入，子CPU准备后同样新准入/项目锁/立即内存与实际PID UUID验证。预计自身4GiB/allocator上限2GiB，既有任务至少8GiB增长及1GiB安全，原mean util≤60/busy≤25%门保持。资格协作25%占空。正式若同卡无其它计算进程，可连续推进；每批NVML查内存、每5秒查进程，一旦共享按至少3×已用时让出且连续饱和时保存让步。无他人吞吐保证，不操作他人进程/功率/时钟。

不设墙钟截止或损失早停。每10000共同步留完整两模型/Adam/RNG/计数及窗口日志定位，唯一80000最终；只对明确资源暂停在完整两模型共同步边界接续。非资源数值失败保留、不自动科学重试；一个seed失败不终止另一个。没有最终部分分数选参/加seed/延长。运行中的日志仅核验前缀/阶段元数据提交，完整原件终态单独归档，全部窗口日志在自有data留hash与大小。

## 全部完成后的固定评价与决定

两seed四唯一最终后原十DEV条件（SYN3源6退化/64帧，ND4源/16帧）全长评价，不作F1内部裁段。九输出：原缓存F0/raw/FB/DIS三个参考，all/output各raw与自身原half-FB.5/DIS.5。自身post估自己的raw场、独立RGB状态，不用F0/GT场偷省部署。raw网络每帧一次、因果/0未来；首帧直通，disable退化原F0不重训练。GT缓存只用于共同指标，PSNR上限120并记录触发、SSIM/edge/flicker/运动残差完整保留，无空域跳样。重复DEV不是独立最终。

完整180序列/8064逐帧/36域seed均值；新两个SL骨干1792图、部署FB1752与DIS1752双向对（合3504/7008calc）、GT438唯一对缓存复用/0新GTflow。六条SL raw/post流120数组放自有data，F0三参考仅引用。此计数先登记后实现；不把全长与D1/F1内部中心均值直接混表。

每层级独立20必要比较：对F0 raw四组要求均值PSNR≥−.05dB、最差序列≥−.50dB且Etemp比≤.90；对原F0 FB/DIS及自身FB/DIS共16组，同质量保护且Etemp≤1.05。另all相对output四组同质量保护且Etemp≤.95，单独检验内部状态增量，不把参数/状态容量不同解释为纯位置因果。all后续资格需自身20＋增量4全过，output只需自身20全过；共44比较，无域/seed混平均。通过只允许另登记完整配对CPU费用/JOD及外部开发，不自动称创新/手机或解封最终。

任一层级失败停止该固定层级/τ8/λ.4/init4/目标/学习率/80K及补救参数网格，不按近门降低阈值、换λ/teacher/clip/宽度/延长救分。即使两者失败，完整已发表基线适配结果仍可引用，不能声称整个IVM或所有递归无效。全部局部正例/取舍保存，四PNG（全均值、配对、44门、固定bear_photons20帧32/video15帧8×两seed全FOV/中心128联系图）实际审阅，PDF另导出。结果/代码/过程/报告分别小步提交并阶段推送；所有旧STOP/原84值/PDF020/封存最终/7号/并行worktree保持，论文目标active未完成。
