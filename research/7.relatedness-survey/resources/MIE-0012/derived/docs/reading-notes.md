# MIE-0012｜FGDPA全文与作者实现阅读

Codex  (sqs_A100) <langxibielangle@qq.com>；2026-11-20。前11项已close才begin12，截止日11月17日不变。0上游代码执行/模型/测试/训练/GPU/Gate3。完整6页v1逐节/全部参考文献阅读；p1–6均原尺寸实际查看（5图、3表、4公式），作者qualitative_results.png实际查看。没有动态视频附件或额外论文supplement入口；不称有时序视觉评估。

## 身份、直接继承与可得材料

v1于2026-06-29T13:59:02Z；PDF/abs为Leshen Zhang、**Ao Li**、Ce Zhu，修正本资源原发现记录An Li；冻结目录不覆盖。原MobileIE p1也为Ao Li/Ce Zhu，确有署名重叠；不能靠A.Li缩写直接认人。arXiv comment/README自述ICME2026录用，未取到独立出版正式版，不补造出版DOI。

p1相关背景、p2§II-A明确following MobileIE并保留FST；p3称同LVW，p4直接TableI/可视化比较，p5基于MobileIE去注意力作baseline，p6ref11明确引文。作者固定代码530c692…（Apr24、早于预印本不奇怪）、tree dbb0a…完整21文件，API逐blob/GitSHA与全包核对；本项依Gate2c审论文-代码，排队13尚未单独开始。README承认partially built upon官方MobileIE，非仅模糊架构相似。

材料含完整模型/工具/训练/loader/loss/config/CLI/logger、90项testlist、1000轮作者日志、23,809B slim checkpoint、5个pyc和一张比较图。原件均完整保留；仅对二进制做SHA/头/ZIP成员大小检查，无pickle反序列化/导入pyc/执行。Apache2 LICENSE文件存在（模板copyright占位），论文abs CC-BY4；第三方UIEB/EUVP和checkpoint来源授权非由此自动证明。

## p1–3：机制/公式/时序

全文引言、Fig1效率图、Fig2完整图/图例实际看。结构是MobileIE式5×5头/PReLU/3×3/FST、body/FST、FGDPA、tail；FST平方交互本来就存在，不作为新时序交互。所有信号当前图，空间频率FFT/DCT不是时间频率，没有历史/未来帧、状态/运动估计/时间监督。纯前向因果不等于无闪烁；运动/曝光/噪声/场景切换未专项测。

§II-B Eq1–2：nonDC3×3固定DCT核零均值/单位范数，DW→PW→tanh(gamma)，并入普通MBR3分支与末PW，理论标准3×3折叠无新增推理核。固定先验只改变训练参数化不保证增加可表示函数集合；论文“collapse减少频率多样性”不能替代精确函数等价证明。默认实现仅MBRConv3加DCT，5×5头无DCT。

实际全读model/uie.py 286行和model/utils.py（至末DropBlock），MBR5/3/1及BN/rawconcat/fusion/slim/FST/FSTS。发现活跃DCTLinearBranch同时PW.weight=0与gamma=0；[单独静态推导](dct-initialization-review.md)说明fresh默认、有限梯度、Adam无外部非零state时，两梯度均零且分支保持无信号。不能从此推断作者历史训练/消融一定用相同源码，更不能否定全部注意力结果；不得为了复现而悄悄替作者改初始化。

§II-C Eq3–4：固定32²池化→FFT magnitude/log1p→GAP/通道均值归一，串接空间GAP→channel attention；另一路原特征max/mean→spatial attention，alpha/beta/lambda静态学得广播相加/相乘，无iFFT。初始化1/1/.5、lambda clamp到[0,1]；“static gating”指参数不是动态路由，不代表attention对输入不变。

文图代码不完全一致：Fig2两attention卷积标3×3，正文写1×1，代码实际1×1；正文先As为C×H×W后又写1×H×W，代码2→C，输出C×H×W。下采样默认可整除时pool到32²，非整除先bilinear缩至整除尺寸，任一边<32直接返回原图，故并非所有输入总是32²。global stats随画幅/视野而变；固定频谱尺寸不证明任意输入分辨率/ROI不变性或时间稳定。

## p3–4：训练与结果

UIEB890中800/90、EUVP无GT1868、UIQS称UIEB无参考子集；没有独立视频划分/clean动态GT。论文RTX3090、Adam beta.9/.999、1e-3、256crop+翻转/旋转、batch4/1000epoch、C12/r4/DCT K4；作者8核图是示意，不能认K8为实际默认。

全读README、config、loader、data/uiedata.py、model入口、loss、main/option/logger及90testnames。配置matching C12/r4/1000/b4、warmupFalse/seed3407，训练硬编码lr.001；data只有随机crop（小图先resize）、RGB/255，未实现论文所述flip/rotation；valid/test原尺寸不裁。testlist90个唯一名可得，但loader仅sorted(listdir)，未读取testlist自动分割，需操作者事先布目录。模型默认original/pretrainedFalse/need_slimFalse，直接README测试命令不自动装载所发slim权重/切换slim结构；strictFalse不证明键/结构一致。requirements.txt不在树，README clone写FGDRA而repo是FGDPA。

loss为signed residual/std/sqrt2的OutlierAware+clip余弦+2PSNRloss，不能只因叫LVW就不核源码；原MobileIE相同类型实现已有出处，非时间loss。main每epoch对valid算clamp后全RGB单图MSE→PSNR平均并挑best，test亦如此，非同训练选优集之外独立最终测试证据。脚本训练启deterministic/TF32默认设置不是已证明全后端重复性。

TableI原尺寸：4075→4234（+3.90%），22.805→23.966（+1.161dB）；FPS718.62→628.25（约慢12.58%），GFLOPs1.1833→1.1796（只约少.313%）。UIQS UIQM和UCIQE分别低于LSNet和Boths、各为第二；不能概括所有数据/指标第一，EUVP两项本法最高。论文同协议声明没有全部baseline运行件/seed/划分哈希；不能与MIE10的不同数据拆分/设备/成本表横拼排名。

Fig3四行鱼/珊瑚/脑珊瑚/柱状体，及Fig4海星实际查看，LSNet部分有明显色块，本法相对去色偏但与GT仍有差异。这是作者选图，不是重评数据/全分布优势或无视频闪烁。作者独立PNG相同四行静态图，不给新动态证据。

## p5：消融与复杂度

TableII baseline18.96是**去注意力骨干**，不是TableI完整MobileIE22.805；+FGDPA4.71dB不能全归FFT。b加DCT+0.32，c仅FGDPA23.67，d去frequency22.81、e全23.97；d比较更贴近频谱增益但仍无重复seed/版本对账，双零实现让DCT消融对应关系待核。K2–8±.1和FFT16/32/>32效果是文中概括，无完整配置/测量表，不能做精确成本收益或无差异显著性结论。Fig5八方向核图已看，默认K4按实现row-major非DC前四。

§III-D：卷积O(NC²HW)，固定32² FFT核部分近似分辨率常数，但downsample/reduction/fullresattention仍O(CHW)，As在代码是C通道，输出乘法/临时张量不是常数内存。4234由slim维度逐项字面算准确相符，不是torch计数/加载权重验证。DCTfold零新增不等于FFTattention可折叠或整个增强零开销；FFT/log/abs/归一/池化与大特征存储/搬运完整成本未被简单GFLOPs充分覆盖。

## p6：部署/结论/出处与日志边界

全文复杂度余段、TableIII、结论/致谢/ref1–26已读。RTX3090在256² .2517G/1142.75FPS、480×640 1.1796G/628.25；Jetson **AGX Orin64GB**640×480/106.9FPS/9.35ms，非前项NX16GB或手机。未给其精度/engine/功耗模式/hostcopy/初稳态/队列/总内存与热稳日志；倒数一致不构成独立测量。不能将固定FFT计算小当手机backend已支持/无fallback，亦不因没部署脚本断言一定不支持。

作者logger原文本1000条连续epoch，Dec29–30，最佳epoch657约23.969094，最后22.979741；不把最后值当best或默认恰好重现表23.966。日志lr.0009与当前源/配置.001不同，未存当时源码/config/weight hash，只有best_slim不能证明源/日志/表完全对应。保留已提供日志的证据，不说作者毫无记录，也不说当前源码训练结果已验证。

无额外新合格身份；参考文献背景不无限扩展全水下/视频领域，原始G1冻结不变。后续应先索引原始训练版本/初始化/权重及split，再考虑部署/时序实验；本阶段不启动Gate3，不改主线/外部代码。对本文的研究假设见analysis，全部124资源综合尚未开始。
