# MIE-0016｜YUV低光论文：逐页、图表与发布工件阅读

2026-11-21；Codex  (sqs_A100) <langxibielangle@qq.com>。在15关闭后才开始16；MIE-0017仍未开始，其同家族工件只作为当前论文的一致性证据。0上游模块执行、0TorchScript/pickle模型加载、0训练、0GPU、0Gate3。

完整读arXiv 2601.17349v1的9物理页/12个编号公式/9图9表及全部参考项；科学页1–8均实际查看原尺寸渲染，p9读参考文字。arXiv原入口写Tech report、v1提交2026-01-24T07:27:54Z；作者主页检索入口称In Submission，固定README承诺接受后发布source，不能用AAAI2026模板版权推断已接收。Crossref精确题名宽检返回10项不同作品；未给本论文推定正式DOI，网页版本追踪见source-version-review。没有另取得独立补充材料。

PDF/arXiv为7作者：Hailong Yan、Shice Liu、Xiangtao Zhang、Lujian Yao、Fengxiang Yang、Jinwei Chen、Bo Li，修索引Feng Yang，保留旧索引字段不改冻结目录。README/主页显示4名不覆盖当前v1的7名。与原MobileIE有Hailong Yan/Xiangtao Zhang共同作者，论文p2 RGB相关工作Yan2025a、p9参考明确引用MobileIE，未将其列入数值表。

|物理页/节|阅读记录与判断边界|
|---|---|
|1 摘要/引言|Y低频亮度与UV高频噪声的分工主张；属于图像域，不是时域建模。机构UESTC/vivo按原文记录，不推MobileIE原文属于vivo。|
|2 图1/相关工作|同一场景RGB/YUV频谱、各通道能量曲线；未给数据集统计/归一化/方差，不能由选例证明所有相机/曝光/白平衡噪声均按此分解。30K“比最轻量现有模型还少”的范围不覆盖其引用的约4K MobileIE。|
|3 图2/式1–3|DSGLA：Y特征maxpool、通道分两路，DDSA膨胀DW+attention与Ghost局部路径；没有时间状态。maxpool并非严格线性低通，噪声抑制须另外验证。|
|4 图3/式4–10|GGA池化空间门控，LAFA融合Y/UV后FFT的幅相池化进MLP通道权重；GI高频DW3/5/7、低频跨通道与LSA。图3使用GT幅相作分析性交换，不是部署时读GT像素的证据。|
|5 图4/5、表1、式11/12、实验设置|GI/FA差分残差；Adam、5000epoch、crop256/b1、lr2e-4、SmoothL1+PSNR，loss系数/schedule/split细节未给完整。表1 LOLv1/v2-real/v2-syn PSNR27.160/31.181/29.081；LPIPS各.131/.104/.051。30K/1.45G为作者报告，无共同重训/独立测量。图4/5为选择的静态放大例。|
|6 图6、表2/3|LSRW Nikon17.99/.525、Huawei21.52/.642；NIQE均值3.550，表中DICM ours3.623并非最低，正文3.535是Zero-DCE行值；RetinexMac DICM3.064更低。MEF3.334/LIME3.594/VV3.126是表内最优，NPE4.071不是。均值优势约.019对3.569，不当统计显著或时间稳定。图6明说每数据集选一图，未看连续视频。|
|7 图7/8、表4–8|颜色拆分消融Y-UV27.16高于R-GB26.65/G-RB26.69/B-GR26.35；仅这个架构/训练条件，不证明所有颜色空间普遍最优。下采样8最佳，删除DSGLA/LAFA/GI及替换模块均有结果；没有参数/预算完全匹配、多seed误差。谱图/Canny特征差仅定性，不是光流对齐或时间一致性。|
|8 图9、表9/限制/结论/参考|GPU6.5ms/CPU124.1ms，未充分给CPU型号、表内输入/精度/线程/全pipeline；后者仅倒数约8.06FPS，不能称30FPS手机视频实证。作者承认其他颜色空间最优性开放。图9静态频谱差不能代替时序评价。|
|9 参考续|MobileIE明确书目信息；LYT-Net/Ghost/频率与HVI融合为当前机制背景，不把所有背景引用自动扩成MobileIE直接被引资源。|

固定作者项目LukasYan30/L3IE-YUV@eaf269b29e7a5d0d0459f6ce616a8c0a0efd5053，tree20f14878，4件297479B全部按Git blob核：README1944B、inference6694B、latency1390B、lsrw.pt287451B。没有trainable model.py/GCSA/FCA/RFFNet源、训练/损失配置、原split/日志、多数据集或消融权重、LICENSE。缺source是README明确未发布，不把它伪称HTTP下载失败。

lsrw.pt是TorchScript ZIP，283成员418559B解压后总量、所有成员CRC通过。只用zipfile/hashlib/pickletools保留清单、metadata指令与97份编译文本56575B，不调用torch.jit.load/pickle.load/作者代码。数据storage126608B不直接当训练参数数目。适用23份自定义图文本（model/GCSA/GhostNetV2/FCA/RFFNet/CA）已完整读，并核pool/up/pad/相关卷积包装；其余标准层包装只清点，不称97件全部逐行审计。编译图不能恢复训练构造、初始化、loss、优化与所有历史版本，manifest保存精确范围。

编译图与论文的具体对账：

- model mangle128第1–65行：输入RGB，额外算RGB max/mean并拼5通道prior，逐RGB混合与偏置构造Y/U/V，分别stem后Y/UV引导融合。并非直接读取手机NV12/NV21/420平面；纸面主要示意未交代这条RGB prior。无时间输入/历史/未来；YUV名称不带来时间状态。
- GCSA mangle67第1–100：交错通道切分，4头的Q/K/V按通道×HW归一化/相关，非HW×HW全空间attention；Ghost分支有池化/卷积/上采样门控。DSGLA pool41是8×8/stride8，up68是固定8倍nearest；inference无相应pad/裁块规则。非8倍尺寸的内部形状兼容只能作条件风险，未运行观察异常。
- FCA mangle80第1–28：完整FFT的abs/angle分别算术均值后进fc；另一路显式转float32做rFFT2，乘可学习fft参数后iRFFT，再结合权重/残差。mangle81的LN/DW乘法及双残差与论文式8的简式需原source说明；不能认定纸面幅相各自调制与发布图完全相同。相位算术均值在±π wrap处的连续性是待验视频问题，不推必然闪烁。
- RFFNet mangle114第1–139、mangle103第1–45：高频两路DW组按16/8/8拆分，乘后先adaptive pool到1×1再送SCA。SCA把[B,C,H,W]换成[B,1,HW,C]，pad最后轴并对[1,3]做im2col，实际跨通道轴，不是原图水平W条带。pool后的SCA输入空间已1×1。这与正文的水平空间条带叙述有具体差异，须作者原source/导出版本解锁；不把图名称当一致性证明。
- CA mangle125第1–30：GELU/深度卷积、decompose差分与sigma缩放，和式12保留非线性；没有能直接折进MobileIE单卷积的证明。全分辨率FFT、注意力统计、通道/空间缓存和色彩转换不能按30K权重忽略。

inference.py 1–169完整读：默认模型lsrw_huawei.pt与包内lsrw.pt名称不同；RGB/255输入，GT同名缺失跳过，未向模型传GT像素。输出clamp、尺寸不同时按GT大小双线性resize，再在0–255浮点上RGB PSNR/逐通道11×11 Gaussian有效SSIM，无Y转换；无LPIPS/NIQE/时间指标或保存图像。autocast分支在非CUDA-fp16时选择CPU context而非null，CPU默认精度语义与GPU选项需后续记录验证，不声称其实际成功或首异常。

latency.py 1–52完整读：脚本设置CUDA_VISIBLE=4，256×256/b1、随机输入、10warmup/100计时、GPU前后同步，no_grad/eval，CPU无线程设定；模型载入和I/O/图像decode/指标在计时外。色彩转换属于图内会被forward计入，不能说脚本排除了全部转换。GPU峰值未reset，不能当指定稳态区间净激活峰值。没有实际运行、使用GPU4或取得本机时间；代码存在不能认证表9数值、精度/设备对应或手机部署。

可供二期的反证/控制：在同资料/质量/预算/输入格式下比较RGB与YUV分工，单列RGB统计prior、色彩转换与编解码/420插值开销；约4K MobileIE须作为适当训练的显式基线，不拼不同协议PSNR。再测运动、曝光/白平衡变化、色噪声和相位wrap/全局统计扰动对时间输出的影响；区分共享跨帧状态与单帧频域模块。当前资料不支持已经抑制视频闪烁或30K等同完整极低内存。训练源/配置/版本/授权/动态GT/手机与完整成本缺口进入证据受限审查，Gate3留下一阶段实验分支。
