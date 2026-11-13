# R74｜显式运动监督的低成本RGB传输检验

登记：2026-11-13；作者：Codex  (sqs_A100)。R73全正常且20门槛全部失败，旧局部STOP保持；当前没有活跃R73任务。R74改变监督信息与传输对象，不继续双矩/rho网格，不预设已有独立新颖性。[先例与边界](../../5.prior-art-survey/docs/55-motion-supervision-and-output-transport-boundaries.md)先于代码/数据评价登记。

执行修订（2026-11-13，正式训练/DEV前）：依[CPU资源执行记录](exploration-r74-cpu-execution-amendment.md)，四个motion模型改为同一CPU FP32后端、Torch2/OpenCV1，顺序完整预算16000更新。复用原正常CPU实际窗资格，core/资格/共享计算不变且所有丢弃更新不继承；CPU模式不要求或声称CUDA训练资格，下面CUDA准入规则仍适用于GPU执行与感知。新attempt002重新冻结资源入口与本修订；原等待001、False工程筛选与明确结束原因保留。方法/资料/目标/优化器/最终选模/下述16质量门槛和质量后的成本规则均不变，不按墙钟或后来GPU可用性中断/迁移CPU训练。

## 固定问题与信息条件

保留同seed R44 square_mse最终累计8000的4075参数部署空间网络，始终eval/frozen，仅对其clipped raw输出进行因果传输。比较相同小型motion网络的两种学习条件：task用最终递归输出对GT的全帧MSE；proxy用正常光GT上固定half-FB估计的运动/接纳代理监督。后者不是传感器运动真值，也不是无监督训练。部署只读取当前/上一raw和上一fused，不读GT、外部flow或未来帧。

这次问题是：显式运动监督是否提供独立作用，并以更低时间处理成本接近原strong recursive .5。不把R73的10%强参考收益门槛改成2%后回写成功；R74事前明确是不同的成本取舍问题，不要求把所有已发表数字挪到同一表。若通过仍只是下一系统设计的有限依据，不能直接当已投稿方法。

## 固定表示和运算

上一raw与fused是两份FP32 RGB状态，1080p为6HW×4=49766400字节，不含当前输入、激活/flow或框架额外内存。原空间网络不改变，避免把继续空间适配的seed偏差误归历史。

对raw/255先ceil avg_pool4，再共享3→8的3×3/PReLU及8→8的3×3/PReLU编码。再pool4得到1/16特征。粗层用固定二维半径4相关（81通道），拼接两特征共97通道→16的3×3/PReLU→3的3×3，输出native单位±64*tanh偏移及置信logit。上采样到1/4，在此用offset/4对上一特征双线性传输；固定半径2相关25通道，与当前/已传输特征及coarse的2通道native flow合43通道，经16通道同型head输出±8残差及logit残差。最终flow范围±72 native像素；不是全运动范围保证。最后flow/logit双线性放大至原尺寸，使用共享连续bilinear gather传输上一fused，越界回退当前。输出为raw+.5*sigmoid(logit)*valid*(warped_fused−raw)。首帧严格返回当前raw，不以GT初始化。没有额外refinement、重置检测、损失混合、网络/搜索半径网格。

粗/细最后卷积均零初始化，两条件初始完全同参数；无BN/dropout，场网络参数<50000（加冻结4075也须明确实际值）。固定偏移相关和一次低分辨率feature warp、一次原生RGB warp均计成本；不能称warp-free、零开销或手机实时。

## 训练与资格

两seed各task/proxy两个模型、每模型新4000，共16000场网络正式更新；从头初始化motion seed7400+seed，空间权重不继承R73。MixedPool原50/50 SYN/real、batch4/T4/crop128，data step8000..11999，不加新像素或打开DEV/最终。Adam1e-4/(.9,.999)/eps1e-8/wd0、global clip1、FP32/noTF32/确定性；两个模型初始参数相同但optimizer独立；batch相同，loss信息条件不同，不假装梯度尺度相同。最终4000唯一选模，每1000只备份。无墙钟截止/科学自动重试，不加LR/宽度/半径/长度/额外seed网格。

task：固定空间raw一次计算可供两条件共享，T4递归不detach，最终clipped RGB对GT全帧MSE。proxy：每个训练窗仅在CPU上对GT相邻帧执行原half_forward_backward，得到native位移和FB接纳；flow双线性缩至1/4但保留native单位，mask用area平均为覆盖率。有效覆盖加权mean(abs(flow−target))/4，加单位权重的logit BCE(覆盖率)，平均全部非首帧/样本。全部监督来自训练GT，不把代理差称真实EPE；不添加task混合或平滑项。空间CNN参数无grad、不在optimizer中、始终eval。

先CPU：已知平移/越界/首帧、奇数尺寸/常量、置信范围、全部有限；两seed实际首训练窗的冻结空间加载/两motion同初始化；每模型2次丢弃更新共8、实际非零flow与confidence梯度、历史梯度及保存/重载≤1e-6检查。0DEV/正式/显卡。资格失败不放宽阈值。CPU正常后才能准备完整训练/评价/汇总/固定图链；全部科学依赖冻结后再按新的共享GPU双采样/锁、完整CPU模型/资料/首窗准备后内部新双采样准入，使用同组4–7、最多4张空闲卡。CUDA实际窗资格丢弃，不继承更新；每seed训练退出即释放GPU，CPU评价不持卡。

## 完整质量、感知与成本判定

仅全部最终权重正常后，既有6 SYN64+4 ND16、两seed共同CPU评价。每learner/seed/scene的motion fields只计算一次，同时运行真实flow和同置信zero-flow两个递归路由；后者不是current，对照中的原生边界接受变化需保留。两个learner×两路由×两seed×十条件=80新质量行/3584逐帧，40motion序列/1752非首帧场预测；直接复用固定空间raw缓存，0新增增强CNN。旧R44 raw/recursive .5的40质量及JOD复用，合120条件/24域-seed均值。逐源完整保留，不能只报跨seed均值。

仅完整CPU后新准入固定ColorVideoVDP0.5.7/standard_fhd/名义30fps/replicate，80新行/3584比较帧。固定十条件×两seed联系表全审阅，帧0/中/末，Input/GT/raw/strong-rec/task/proxy/proxy-zero-flow；不选片/曝光修正，不当整段或MOS。

proxy真实flow raw对原raw、同权重zero-flow、独立task真实flow，各两域×两seed均要求PSNR差≥−.05dB/JOD差≥−.02/Etemp比≤.95；对旧R44 strong-rec .5要求同质量保护且Etemp比≤1.02。每项最差逐条件PSNR差≥−.30dB，合16联合门槛。任一失败即停止本motion-supervision组合的半径/宽度/监督权重/代理来源/长度等局部搜索，不换GT教师或调confidence阈值追分，不把proxy再加外部flow当主输出。

只有这些质量门槛全部通过，才测一次固定成本：同一已加载raw序列、相同CPU/线程、含tensor转换/预测field/完整原生warp与接纳，与原half-FB+warp对照；每条件2预热/3重复，不含文件读写和冻结空间CNN，但二者共享该部分且须明说是时间处理阶段。要求两seed条件中位数成本≤.5×传统阶段、参数/状态符合上述限额，报告每条件分布而非仅均值；这仍非完整手机实测。未通过不扫精度/后端/输入尺寸以挽救。通过才考虑独立外部与设备，保留最终素材仍不打开。

代码、资格、原完整结果、解释、实际图审阅、过程、报告分别小步提交，阶段推送；指定作者/no-ff/自有data/project-python保持，不改移动演示。创新、可投稿方法、自然动态配对、手机和独立最终仍未建立，远期目标active。
