# MIE-0022｜JND轻量神经预滤波：正文、实现与编码口径

2026-11-21；Codex  (sqs_A100) <langxibielangle@qq.com>。有效序25，FastSHADE关闭后开始。当前逐视频补充结果获取尚待收尾，未作Gate2决定。0研究源码执行/模型加载/训练/GPU/Gate3；公开分享网页导航不执行研究模型。

## 版本与真实阅读范围

当前arxiv2510.10648v3为2026-11-08五页，作者Yangyang Wu、Chenlong He、Yuanheng Chen、Yibo Fan（复旦），前两位同等贡献。官方摘要comments为Submitted to ISCAS2027，不是已录用；仓库results.md仍写ISCAS2026 submission，保持来源时间区别。v1（2025-10-12）七作者、v2（2025-10-18）五作者均另存完整PDF和衍生页，不解释名单变更原因。v3五页含33参考已全部阅读全文、科学p1–4的4图6表实际查看。v1/v2→v3全部文本diff读，v1p1/p3与v2p1实际看；不声称旧版每页视觉审查。

p3明确以MobileIE[33]为backbone，采用MBR/FST/HDPA，把5×5改3×3并将head/body收窄后tail扩展；p5[33]精确指向原论文2507.01838。v1已用同书目/架构，只把名称误写MobileNetIE，v2/v3改MobileIE，不是v3才新增引用。该文是直接结构/训练框架继承先例。

作者链接viplab-fudan/FJNDF-Pytorch，固定commit3b46739e4f524c205c9c277f9c4b1673b9d5aedc（2026-11-08）、tree8164518198b68bb06ddd25c0cacc8c67c57f34a9。ZIP150文件2121784B全部Git blob/长度/SHA匹配并独立可解析留存；128文本准备不等于全部逐行精读。当前核心模型/loss/配置、训练验证/数据及推理/编码BD链实际静态读，通用工具仅相关函数，未用helper和历史经典模型非所有实现逐行重审。3PTH只CRC/操作码，未torch.load或反序列化。独立项目MIE0023尚未begin，当前作者源核查是本论文适用实现对照。

## p1–2：JND、基准与任务边界

目的是在编码器前压缩人眼不敏感的空间内容，输出仍要送codec；不是低光增强真值恢复，也不以静态滤波改善代替时间稳定。平台集成DCT域CSF/LA/CM/SA阈值与注入方式、四类codec和六指标。所谓JND阈值依赖建模/观看条件，不是对任意手机显示/观察者/噪声运动的可见闪烁阈值认证；本论文不提供视频MOS或新主观实验。

Eq1乘法融合阈值，Eq2广义系数抑制/高斯滤波为传统teacher机制。本文选[13]+[6]（TIP2016阈值+2024分类注入）生成2008对DIV2K/KonJND训练图。TableII各数据/指标的最好组合不同，HEVC四指标ALL下[5]+[6]的-.78比所选-.70更低；“top-performing reference”不等唯一全数据最优。Teacher GT是空间预滤波的伪目标，既不等干净低光GT，也不等真实视频时间标签。

仓库`results/results.md`明确当前结果只用于image coding，旧方法的temporal、显著区域detail enhancement和QP调整未纳入；因此与原完整竞争pipeline要分开。TM注册类只是返回ones的placeholder，不能因2009模型名含时空或registry有TM就声称已实现时间感知。当前默认teacher没有历史/未来/时间状态。

## p3：小网络与混合损失

核心是原MobileIE风格FST平方非线性与双路径attention、空间卷积，改为预测加性残差并与输入相加，非直接照搬原RGB Retinex除法。当前Lite默认只处理Y：3通道YUV拆Y为1通道进网，UV原样拼回。head8/body8/tail12，训练rep4；FST使用同一个block输出的两个标量乘积/平方加bias；attention是当前帧全局平均通道门控和局部通道最大门控，没有跨帧state。全帧池化可把当前全局变化影响局部，但并未测得实际闪烁。

论文Lc(Charbonnier)+.16Lm(MS-SSIM)+.02Lf，Lf包含与teacher的全DCT平方残差、与原图的单边幅度约束。LF区（zigzag前10/64）惩罚输出幅度减小；HF区惩罚幅度增加。它是软约束，不禁止LF增强/HF减弱，也不约束系数符号/相位、相邻帧或连续观看的可见程度。“physical conservation”是作者命名，不能自动当物理能量守恒或真实时间一致性保证。

当前源码8×8非重叠DCT在N=8时为正交归一化，全部系数等权平方残差按Parseval等价于覆盖像素的平方残差能量；不是额外频带选择信息，不能把整个组合loss都称无效/冗余（它还含单边constraint，且空间项是Charbonnier/MS-SSIM）。loss.unfold对非8整除边缘只覆盖完整块，默认224训练patch可整除；不等推理丢边缘。

需保留论文-代码区别：Eq6是正部的平方，`FrequencyConservationConstraintLoss`实际对正部直接求和，没有平方，梯度形状不同。`_zigzag_indices`沿反对角线不交替反转，但默认K10恰为完整前四条对角线，低频集合与标准zigzag相同；只在其它切到半条对角线的K可能不同。6D掩码对5D系数多出前导singleton，不改变默认mean，不称批次混合或运行错误。

训练器仅对pred执行softclip01再进入MS-SSIM；k2时输入0/1分别映射约.119/.881，整个区间都变，非仅超界clip，target不做同变换。其它loss比较未softclip的Y。论文没有明说该变换，不把它省略后称完全相同训练。仅源码条件分析，无训练验证。

## 参数、折叠和导出范围

字面定义核算Lite训练24949、折叠1861参数，与24.95K/1.86K相符，拆项保留。`eval()`另建slim但保留注册的training network；统计脚本sum(model.parameters())会包含两套，默认26810而不是1861，且原样数字后标K未除1000。这个脚本范围问题不否认孤立部署拓扑1861；没有实际调用计数或加载权重。

当前baseline默认1Y、C12、5×5首层的源字面计数41413train/3257slim，与论文表38.92K/3.06K不符。发布权重首conv的操作码shape字面48×1×5×5，Lite32×1×3×3，与当前source分别一致。若假想uniform12全部3×3，算术38917/3065会接近表值，但只是待解版本假设，不替作者修改原文或断言训练历史。文件字节数/原Storage含BN统计不等学习参数数。

当前slim只折MBR/BN，FST/动态池化/门控/残差仍在；不能把全网当单Conv。默认fp32，半精度选项中创建的新slim为默认dtype并只to(device)，需独立查导出/输入精度等价，未运行也不宣称默认FP32失败。7.15GFLOPs是作者数字，14.1%分母50.70为IQNet，不是MobileIE；41%是相对论文12.15基线，当前baseline版本差异须先对账。

## p4：训练/数据与划分

原文Adam3e-4/100Kiter/b16/224patch，λ1/.16/.02，K10。当前yaml补充seed123、weightdecay1e-6、cosine97000、warmup3000、split_num1/index1、dataset_enlarge4、center_crop224及paired h/v/rot90；rot90值1在实现中作概率1。替换random_crop是另一个路径，内部循环覆盖坐标变量i，会使pair使用0/1不同纵坐标；不是当前center_crop失败，不能以此否定已报实验。

2008训练CSV和codec同集2008条可得，四外测CSV为HEVC5、XIPH15、MCLJCV30、MCLJCI50，均8bit420。元数据不等取回全部图像/序列，不证明语义无重叠。数据文档一处写KonIQ-10k，论文/README/同文档引用写KonJND-1K，实际原始来源/许可与具体筛选仍有歧义。原/teacher同basename位于不同root，不是同路径GT泄漏。split生成器是行随机而非scene/clip分组，README例5/.9/.05/.05/seed42与脚本默认10/.8/.1/.1、训练seed123分开；没有反序列化splitPKL或冒称验证实际成员。

GeneralFRDataset对YUV读取/上采样/归一化再PIL，训练只用Y；不把这些平面当RGB色彩转换。验证以teacher为参照、uint8 clip后PSNR/SSIM及总loss选best，各指标best可来自不同迭代；不等编码BD的best，也不等原图/人眼GT。模型eval/再train在每张验证图重建/丢弃slim，预测/输入/ref列表留device内存，验证开销不等单网forward。默认同一split不是五折平均证明。

## p4与完整表：评价、公平性和ALL

所有本文实验是All-Intra、QP27/32/37/42、仅滤亮度。四数据集上VVenC TableIII的六指标ALL为-2.80/-4.12/-2.83/-4.72；它是不同指标各自BD百分比再平均，不是人眼统一分数或所有单项都省码率。例如HEVC PSNR+.87和MCLJCI+2.74仍比原anchor需要更多率。x264的PSNR原方法[5]-2.00比Ours-1.51更低，不能写每个指标总最优。

TableIV/V只四指标，Ours HEVC -.34/-5.70/-4.71/-6.20均值-4.24，TableV .87/-4.20/-2.90/-3.67均值-2.47，和TableIII六指标-2.80不是矛盾。当前repo16组表均六指标；32个baseline/Lite行抽出阅读。当前results.iccv_yan_2025的.69/-4.43/-3.34/-4.18对应TableV BaseLine+Lf，不是只spatial的BaseLine；不同训练条件不能凭名称混成同一行。

TableV baseline已比teacher的感知指标更省，但PSNR+3.52更差；加Lf使PSNR+.69、VMAFNEG-4.18，MS-SSIM反而-3.34较无Lf-3.50差，不能说每项都改善。Lite降低成本同时牺牲该配置部分感知收益，无seed/CI或独立主观实验。

原文CPU XeonGold6230、GPU RTX2080Ti的.428s/10.88ms是作者网络计时，非手机、非完整编码链/热/功耗/峰值内存测量。条件倒数约2.34/91.91，不是测得视频FPS。7.15GFLOPs/1.86K与单RGB原MobileIE预算、设备和任务不能直接拼成同等胜出。

## 当前推理与编码链实际读到的边界

单文件`filt_one_video`每帧task加载模型/权重；批处理`filt_one_dataset`每个video加载一次，再循环帧复用模型。这两入口不能概括成都逐帧重载，也均无时间state。输入pad32/crop，YUV按bitdepth全量程归一化，UVrepeat/抽样，保存clamp+round，与验证uint8截断和loss softclip分开。

批处理load_video_params对Y4M/image分支continue导致未加入任务；单文件有Y4M路径，不说整个库不支持Y4M。`--target`不改变实际由cfg指定的self.target。当前工具是离线读写/拼接，不自动成为流式pipeline；这些是源码条件，不是实跑失败。

编码默认FRAME_NUM=1、CQP四QP，故虽vvenc/x264/x265配置GOP/周期200，当前只首帧仍可作静态intra对比；扩大帧数不自动维持All-Intra。编码器明assert8bit420，VMAF hardcoded8/420在该默认范围相符，不误判所有默认失败。PSNR/SSIM用FFmpeg All汇总、HVSM/MS-SSIM裁8/16整除边缘、VMAF两个指定model，必须保留不同度量口径。原图路径自动改回ori供REC对比，而非拿filtered当参照。

calbdrate对每图/视频按共同QP、去非有限且至少四点，Akima计算；overall是先按QP平均率/质量再算BD，不是每序列BD平均。overall两侧独立剔除缺失、未强制共同视频集合，缺测情况下可能人群不一致；没有raw结果就不能认证当前是否发生。异常返回NaN/缺文件skip均需未来完整性检查。未执行任何编码器/外部model/metrics。

## 二期价值、缺口与可证伪控制

有价值的是极小网络+伪teacher+参考独立软约束与明确编码链的先例，可启发训练目标与“保结构/抑高频”预算分析；它没有时间可见性或去闪烁证据。简单DCT/感知loss+MobileIE已有直接先例，不能以此组合本身主张创新。

后续控制包括：同数据/teacher/容量/precision与完整成本对账，线性/平方constraint和等权MSE分离，符号/相位/曝光及真实运动噪声、固定时间参照/无未来输入、空间质量和连续时间主观/客观指标、真正跨帧/All-Intra与inter编码分别评价。先对齐纸面与公开baseline/训练/导出版本，再作反证实验；全部留待授权实验分支。本项逐视频补充CSV获取、正式发表及原split/数据/权重历史、运行/时序/手机成本缺口在Gate2前统一审查，不提前综合。
