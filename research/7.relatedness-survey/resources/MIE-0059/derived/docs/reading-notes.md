# MIE-0059｜ULightIF：证据可得性与关联实现精读

2026-11-21；Codex  (sqs_A100) <langxibielangle@qq.com>。有效队列分析序21，静态ID59保留。20先close后才begin59；不是按数字ID跳号补读。0作者源执行/反序列化/模型加载/训练/GPU/Gate3。

## 身份、正式版本与获取边界

Muhang Cheng、Haiyan Huang、Hongwei Mo；DOI10.1109/TMM.2026.3723605、IEEE文号11655203。当前出版者HTML内静态JSON及Crossref核为TMM28、13430–13441（12页），出版/插入日期2026-08-13。当前isEarlyAccess=false、isOpenAccess=false；原README和网页旧条目1–13页保留为旧元数据，不能把两者都当读到的全文版本。

**正式正文未取得，不能声称全文精读完成。** 三轮实际获取：出版者正常PDF/stamp/元数据pdfPath/注册VOR/参考入口，Crossref、OpenAlex、S2，题名/PDF/arxiv/作者网页搜索及官方作者项目。202空响应是无正文；部分HTTP200实际是登录/摘要HTML；注册VOR也是HTML，不按URL.pdf或success认全文。OpenAlex标closed无PDF，S2的openAccessPdf指向DOI并写CCBY，与IEEE当前非OA冲突，不据此认定可下载PDF/许可。所有原回执及本文17来源清单留档，不绕过认证或把检索未命中当全文不存在。

S2当前46条参考有精确MobileIE正式DOI/arxiv/题名，IEEE元数据47参考；只核了书目级直接被引，未获文内引用位置/语境。官方作者README则明确ack主要建立于MobileIE，足以支持实质代码关联。不可把这两类证据混成已核正式正文引用段。

出版者摘要报告：低光IR/可见光无监督融合，用先增强后融合结果指导先融合后增强，包含增强/融合/分解三网络及卷积重参数化，约.0597M参数；无需正常照明GT但需要成对IR/可见光。这些为摘要级作者主张，没有取得完整方法公式、训练/数据协议、表图/消融/质量/下游与成本数值。不得补写不存在的页节笔记或排名。

## 实际可读材料、版本对应与视觉

作者cmuhang/ULightIF固定bc952b6f1f18a6480628fc91f6ce96a93484817c，tree14355b201e2309299f61a175211b9b0c8f4d2648，提交2026-09-02；18件1025443B逐Gitblob/原ZIP核。本文把它作为关联源，不begin下一项MIE-0114。

14文本（README+13py）留档；实际全文读README、build/build_slim/base_model/net、两test、train/options、paired_data、image_utils、losses、train_utils共13适用文本。utils/schedulers.py仅查类/函数清单和全包导入：未被本发布调用，实际优化器使用torch.StepLR；不伪称其430行逐行审计。三weights全部ZIP CRC、FloatStorage/操作码检查，不执行GLOBAL/REDUCE或torch.load。

唯一framework.png已原图实际查看：同对IR/VIS的两种处理顺序及约束示意，不是视频/真实时序样例；没有可得的论文其它图表、原始对照输出/GT。框图不能替代全部科学图表审阅。

README明确三份weights为原论文test_p.py使用，Lightning重构训练产物供test.py，不能任意互换或称重构重新训练已复现原文。test_p内直接定义三slim模型，test.py加载三个Lightning checkpoint并折叠；原权重三个小pth存在，重构的last.ckpt没有随包发布。

## 当前代码机制、信息与参数

IE默认24通道，3×3Conv/PReLU/残差及sigmoid单通道illum，Ref=clamp(VIS/Ill,1e-4,1)。IF默认16通道，共享encoder分别处理VIS亮度、增强VIS亮度和IR亮度，两个decoder调用得到fu、fuEn；RGB由原VIS/增强VIS的CrCb配回并clamp。ID默认32通道处理fu，输出sigmoid RGB反射R与单通道L。所有都是当前配准图像的空间操作，无历史/未来、时间状态、flow/对齐估计或scene-cut协议。

IR是部署输入，并非仅训练teacher；不能把该融合质量当同RGB信息预算的MobileIE视频结果。文件同名配对不证明物理配准/同步。RGB/YCrCb转换是当前软件步骤，不是原生传感器YUV或零拷贝。

源形状算术与三个检查点FloatStorage总量分别为IE6145/24580B、IF14305/57220B、ID39204/156816B，合59654≈.0597M，支持摘要参数数量级。这里是结构/存储核验，不是模型运行计数或性能复现。

test_p实际forward执行IE、IF三个encoder/两个decoder分支、ID，并保存viEn/fu/fuEn/fuR/fuL；timer包括这条完整网络路径和转换，预读图/拷贝调用与随后保存minmax/量化不在timer内。test_p首次10次warmup，test.py当前没有同类显式warmup；不能混合两入口时延。

源码末注建议常规fu只统计IF，低光fuR统计IF+ID（53509参数），但脚本默认总计三网络和执行全部分支。固定无状态slim权重下，fu本身只依赖VIS/IR encoder，fuEn才依赖IE/增强VIS；fuR=ID(fu)。因此裁去训练teacher辅助路径是静态依赖推断，**本次未改代码、未测等价/速度**，不可把53509参数与未裁剪三网络时延混报。共享权重多次执行也不能只按一遍encoder成本计。

count_flops把thop返回的MAC/G直接标GFLOPs，未给2×口径或完整非卷积覆盖；默认640×480统计不代表全部测试分辨率。没有手机、精度/热稳态/完整内存功耗原始记录。MBR采用reflect边界和BN running统计合并到slim，同图PReLU仍保留；存在折叠检查函数/示例只表示作者提供检查，不是本次执行通过。

## 训练、损失、推理协议及视频问题

Lightning重构分IE→IF→ID三阶段；默认每阶段10epoch、batch8、crop96、Adam1e-4、StepLR50/.5、seed3407。训练只fit train_loader，save_last、无真实validationloader或best质量选择；options列出的旧last.ckpt路径不是实际随包材料。不能把这些发布默认值称原论文训练设置。

IE损失用输入平均亮度构造.6曝光目标、空间邻域梯度和illum平滑；IF分别对fu/fuEn施加强度max、Sobel梯度max和48窗口SSIM约束；ID以fuEn为局部亮度/感知参考、R*L重建及棋盘下采样一致性。ID训练还加固定sigma15/255合成高斯噪声，验证函数不加；不是时间邻帧一致性或真实噪声GT。

ID感知项使用冻结ImageNet VGG19，把图转灰三通道输入，只取后层特征；代码中无显式ImageNet均值方差归一化。无需正常照明GT不等于无外部预训练先验。VGG属训练/重构加载路径，不在原test_p三slim forward参数预算内；本次未下载/执行VGG。

PairDataset同名IR/VIS联合crop/翻转/旋转，原图需容纳crop，且目录清单无扩展过滤；没有序列/时间划分。MSRS/LLVIP/M3FD只给官方链接，数据本体/许可/实际训练划分和预训练重叠未对账，本阶段不下载数据或查看项目封存最终图。

**保存不是简单clamp**：ImgRGB_save/ImgGray_save每张输出按全张量min/max重新拉伸到0..255，无epsilon。batch1时某个极值变化可以影响全图的输出映射，恒值输出有零分母的条件算术情形；这是可证伪的后处理时间风险，未观测真实视频闪烁/NaN或异常。即便网络稳定，文件后处理也需要独立控制；不能将保存前tensor、归一化保存图与时延口径混在一起。

test_p CPU/CUDA自动选、weights_only加载原pth、只按VIS文件名查IR，不检查配准/时间；test.py载重构ckpt且check_fold默认打印合成随机输入差异，无阈值assert，IF还经RGB转回Y混入clamp/色彩变换影响，不是单独卷积代数等价认证。options的布尔开关未指定bool转换，传字串False仍可能为真；仅静态入口限定，未运行首异常。PLIFModel.forward若直接给RGB会落到单Y通道原IF网，但实际training_step显式转Y，ID阶段用slim内部转Y；不将未用helper问题倒说训练必败。

## 结论边界、范围线索与交接

它提供明确MobileIE派生、跨处理顺序teacher与训练/部署辅助分支分离的先例。二期可据此检查训练辅助信息与推理因果输入、网络/颜色转换/保存后处理三类时间扰动；需要同RGB信息、同质量/预算、真实运动曝光噪声和完整手机成本控制。IR额外信息、不同数据/训练以及没有时间机制都限制直接迁移，不能把当前双传感器成绩当视频方法对照胜出。

README另列SpikeVFuse/FDFuse/LEFuse，旧G1A0001已存同书目；定向检索只得到ULightIF共列，未得具体MobileIE直接引用/复用/视频接续证据。按冻结范围不无限递归到融合领域，不把“没新增合格证据”写“这些论文没有引用MobileIE”。若以后获得合格直接关系即做Gate1增补；本次未新增ID，不修改Gate/目录。

建议证据受限结项：正式全文/补充/完整图表及确切引用语境、原训练/数据/消融和数字协议、历史与重构权重对应、手机/时序证据仍缺。当前只支持出版身份/摘要主张、书目级被引和作者显式代码继承、当前工件机制及条件推断。后续合法全文或原实验材料出现另追加复审，不能回写本轮冒称全文已读；Gate3预实验留给新任务分支。
