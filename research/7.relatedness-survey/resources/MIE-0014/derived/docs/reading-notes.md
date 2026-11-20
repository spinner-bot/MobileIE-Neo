# MIE-0014 DRNet：全文与适用作者实现阅读记录

Codex  (sqs_A100) <langxibielangle@qq.com>；2026-11-20全文/源码阅读，2026-11-21恢复后的对账记录。0上游源码执行、0模型加载、0训练、0GPU、0Gate3。MIE-0015仍未开始；此处作者实现仅服务当前论文一致性审查。

实际原文为arXiv 2605.08627v1，13物理页；不是已取得的TMM正式全文。作者Ao Li（索引曾误An）、Xiaoning Liu、Sheng Li、Yapeng Du、Zhen Long、Lei Luo、Le Zhang、Ce Zhu，与MobileIE的Ao/Le/Ce重叠。正式DOI和TMM28:12144–12157通过Crossref核验，年份2026，未将默认Jan1或模板页眉2021当发表日。获取过程见publication-acquisition-notes及report-time-acquisition-review；其中早期“读到1–6、8页”是历史进度，后由code-initial-audit完整阅读范围接续，不删除旧记录。

全文13页所有节/75条参考项已阅读；11图/8表逐项检查。原科学页1–12均在11月20日查看原尺寸渲染，参考页13读文字；11月21日再次实际查看p10/p11。纸面图片与特征图是作者选择的静态例，不能当连续视频观察或独立重现。原件、全文文本、所有页图入口分别见sources/manifest、paper-assets和figures目录。

|物理页/节|实际阅读与证据边界|
|---|---|
|1 摘要/引言、图1|多任务恢复、会话初始化配置主张；图1参数/平均PSNR为作者比较，不是手机测试。|
|2 引言/相关工作、图2|残差频谱是静态退化示意。结构重参数化段明确MobileIE[40]并述并行卷积/BN，不是将MobileIE列入视频对照。|
|3 总结构/§III-A、图3、式1|单张RGB图，四层hourglass；任务字符串映射固定one-hot，不是语言模型或逐帧学习估计器。图内已写DWT/IDWT。|
|4 §III-A/B、图4、式2–6|双bank加权仿射，中间GELU；TSM共享两Linear后分两Softmax。盲模式在文字中是额外generic one-hot任务。按固定先验分别折每个bank，不跨GELU折成一个线性层。|
|5 §III-B/C/IV-A/B|prior-related计算仅在配置时；SW-SA保留。CWTE名称为Continuous，实际明写离散Haar分解。3×3后LL递归、高频跳接。BSD400/WED、Rain100L、OTS、GoPro、LOL训练，BSD68/Urban100/Kodak等评价；128裁块、150epoch、batch32、Adam2e-4、L1和翻转旋转。|
|6 表I/II|五任务先验模型30.45/.914，盲29.62/.903；三任务32.69与AdaIR平均值相同，非每列第一。LOL22.30、GoPro29.01不是表中全部最优。任务信息与多任务预算不可混成公平单一上限。|
|7 图5|五任务RGB放大和误差热图；文字“更优”是作者定性解释，缺独立原像素/动态GT核对。|
|8 表III、图6、§IV-D/E|同一混合退化图多次恢复，不是相邻帧递归。CDD-11 haze/rain再加sigma25；盲多次对三重19.61少.66dB，单次14.56；双重盲多次17.85反低单次19.07。Fig6最终18.76/15.80为所选例，与总体表均值不是同量。|
|9 表IV/V/VI、图7|真实SIDD/RTTS/RealRain，FADE/NIMA不是配对时序误差；SIDD三任务28.23/.602、五任务26.72/.758。四空间分块噪声实验是空间非均匀，不是曝光随时间变化；“unseen”集合含0且seen也含0，不能照抄所有噪声级均全未见。表VI静态去噪比较不是动态视频。|
|10 §IV-H/I、图8、表VII/VIII|128×128、单RTX5090，作者融合前16.49M/34.59G/47.78ms，后7.39M/16.48G/28.92ms；内存作者正文单位MB，表头M。未给手机/首帧/切换/全pipeline协议。正文30.47、表I/VII30.45、表VIII30.24各自保留。去CWTE同时变容量，去DRMLP同时变训练参数；不能作纯单变量归因。图8正确任务提示更亮，非自动识别或时间稳定证据。|
|11 图9/10、消融|各任务子带/分支激活与权重静态可视化，色图归一化细节不足；只能支持展示的模式差别，不能单凭图推出因果知识共享或优化冲突已解决。|
|12 图11/结论/参考|融合权重相似度是描述性分析，.55/.94不独立证明迁移；作者明确atomic prior、组合提示单次恢复留未来。ref24 PASTA触发G1-A0002，现已纳入MIE-0125；不在此提前深入分析它。|
|13 参考29–75|MobileIE为ref40；一般Swin/PromptIR/频域参考用于解释当前机制，不无限扩展成MobileIE直接被引。|

作者代码固定f1d25d2acc2db0f16e1b8fc0b687bb9eaf7a1e4d（2026-06-07），tree d914357cd3e2f25cdfff65e6815bdef370c80436。21件共152472B全部按Git blob对账，原件与本地可读路径见author-source-manifest；未安装其中requirements或执行入口。

适用核心已实际读：README全文；DRNet_arch 77–589及相关导入/初始化；net_utils全文；train/options/transfer/inference.sh/requirements；dataset_utils/degradation_utils全文；image_utils裁块与增强；scheduler实际warmup-cos类；inference_5tasks全文和3tasks完整差异；metrics compute_psnr_ssim。未作完整审计的范围：trunc-normal通用工具1–76、未用的matlab/tensor/dtype/通用工具、custom-autograd数值正确性、环境/历史实际训练状态。当前末尾main参数计数只是上游自测入口，未运行。源码已归档不等于21文件全部逐行人工审计。

关键源码锚与检查：

- DRNet_arch 77–172：TSM两Linear→分Softmax；DRMLP双专家bank/中间GELU。switch_to_deploy在deploy时立即返回，首次计算权重后detach并删bank/TSM，不能在同一已转换对象切换新任务。固定先验折叠为符号线性关系，不是已验证FP32/FP16数值等价。
- DRNet_arch 332–348、488–582与net_utils：离散Haar分支，四尺度mask和位置索引每forward重算；mask在CPU构造后搬设备，SW-SA、LN、GELU、wavelet和拷贝仍存在。forward只取当前图/任务，没有时间state、历史或未来帧。
- dataset_utils 36–51：set→list决定one-hot次序；inference5 243–250固定denoise/dehaze/derain/deblur/enhance，inference3 245–252固定前三。训练语义映射未随checkpoint明示；这是条件兼容风险，不推断作者历史结果必错。
- train 18/94–103/116–124、options 8–14/33、README 60–67：5维任务硬编码；默认de_type只有三种base，若不改参数存在维数不兼容风险。README暴露2GPU但num_gpus4；150epoch和b8×4意图与options1200/b2、scheduler max100分开；warmup15后cosine在100之后没有自动被截成常数。
- dataset_utils 67–135/169–185：不同任务重复倍数3/120/5/20改变混合分布，缺外部ID列表和实物dataset，不访问源码中的他人绝对路径。de_type单独选denoise25时_merge_ids仅检查denoise15的潜在配置风险，不把它冒称默认首次报错。
- inference5 25–40/145–179/224–264及inference3对应：RGB/255，AWGN在像素域加后clip和uint8；denoise先裁32倍数，镜像pad到下一个64倍数（已整除也+64），输出再unpad。5任务sigma硬编码25、3任务50，CLI无sigma，README使用不存在的inference.py且宣称可指定sigma。
- metrics/psnr_ssim 10–29：clamp预测/GT至0–1，RGB逐图skimage、data_range1，无Y转换或边界裁剪。不能用同目录其他未调用metric函数替换实际口径。
- README称MIT但21件没有LICENSE；承认源自PromptIR，未见MobileIE模型代码直接复用的证明。不能从共同作者/同类重参数化推出同一实现。

原文额外generic盲先验未在公开3/5任务入口里找到对应独立配置；训练/推理名单只有原子任务，不能据此复现w/o prompt列。作者权重已于本次恢复后按修复的获取器新label重取，获取状态及静态容器检查另记；旧截断文件和错误success回执保留。数值结果、数据重叠/许可/精确split、历史语义映射、正式版差异、手机/完整成本与视频闪烁均待证。
