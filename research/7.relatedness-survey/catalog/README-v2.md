# 永久索引与当前初级队列 v2

2026-11-20；Codex  (sqs_A100)。124项：A53/B29/C42。分级和概括仅为Gate1后的初整理，所有资源均需逐项Gate2。E不参与排序；不表示方法有效或已读。

原ID不变，新增MIE-0114…0124。已开始1–4保持前缀；原MIE-0059据新增官方MobileIE复用证据重新初评90分。旧40分及旧顺序见[index v1](index.json)；理由见[评分记录](scoring-v2.json)和[队列决策](../docs/0006-增补索引与队列决策-26112016.md)。

状态以[顺序日志](analysis-log.jsonl)及各项审查为准，此表不是完成清单。AnyMod正式版复审未关闭前，MIE-0004和后项仍禁止推进。

|当前顺序|永久ID|初级分级|类型|资源与入口|初级用途|
|---|---|---|---|---|---|
|1|MIE-0001|A / 100|paper|[MobileIE: An Extremely Lightweight and Effective ConvNet for Real-Time Image Enhancement on Mobile Devices](../resources/MIE-0001/identity.json)|MobileIE原始论文与补充材料。先校准架构、训练及GPU/手机效率口径，为所有引用与派生实现建立事实基准。|
|2|MIE-0002|A / 100|code|[AVC2-UESTC/MobileIE](../resources/MIE-0002/identity.json)|官方固定提交实现。校准训练/重参数化/损失与实际推理路径，并与论文逐点对照。|
|3|MIE-0003|A / 100|paper|[AnyMod-LLVE: Low-Light Video Enhancement with Modality-Agnostic Inference](../resources/MIE-0003/identity.json)|明确的低光视频增强及模态无关推理。直接影响视频信息条件、对照选择与创新边界，列最高优先级。|
|4|MIE-0004|A / 100|code|[lhfgghc/LLVE-AMNet](../resources/MIE-0004/identity.json)|AnyMod-LLVE官方实现名与论文名不同。核查真实输入、历史/未来帧、状态及发布配置。|
|5|MIE-0005|A / 100|code|[lhfgghc/LLVE-AMNet-NPU](../resources/MIE-0005/identity.json)|官方README链接的NPU版本。重点核部署路径及其与论文模型的信息/计算条件是否一致。|
|6|MIE-0006|A / 100|code|[nlsidf/vnrit](../resources/MIE-0006/identity.json)|公开Rust代码同时提及MobileIE启发和时域/运动/闪烁声明。直接核声明是否对应神经模型及实际状态路径。|
|7|MIE-0007|A / 100|code|[nlsidf/ndk-vnrit](../resources/MIE-0007/identity.json)|相关NDK版本。核查与Rust上游、Android封装及所谓抗闪烁机制的实质关系。|
|8|MIE-0008|A / 100|code|[KrakeCZ/AIchat-Android-Local-AI-Agent-](../resources/MIE-0008/identity.json)|Android插件菜单中出现MobileIE/TFLite与时域去噪。需分清注册/菜单、可运行模型及真正组合路径。|
|9|MIE-0009|A / 90|code|[tediyankov/reef-track](../resources/MIE-0009/identity.json)|水下跟踪代码含MobileIE增强实现。核是否逐帧、是否有增强状态，以及跟踪效果能否说明闪烁。|
|10|MIE-0010|A / 90|paper|[Advancing Visual Reliability: Color-Accurate Underwater Image Enhancement for Real-Time Underwater Missions](../resources/MIE-0010/identity.json)|TeleAI署名的实时水下增强。已有明确MobileIE关联，是工业接续、轻量结构和重复风险的重要核查项。|
|11|MIE-0011|A / 90|code|[Cloudyu1215/UIE](../resources/MIE-0011/identity.json)|TeleAI相关水下增强的作者实现。核MobileIE联系、结构与测试/部署条件。|
|12|MIE-0012|A / 90|paper|[Real-Time Underwater Image Enhancement via Frequency-Guided Dual-Path Attention](../resources/MIE-0012/identity.json)|同组频率引导双路径注意力水下增强。重点核查原MobileIE组件如何接续以及部署成本口径。|
|13|MIE-0013|A / 90|code|[LethyZhang/FGDPA](../resources/MIE-0013/identity.json)|同组水下增强代码。核结构重参数化与频率注意力的实现/部署边界。|
|14|MIE-0014|A / 90|paper|[DRNet: All-in-One Image Restoration via Prior-Guided Dynamic Reparameterization](../resources/MIE-0014/identity.json)|同组先验引导动态重参数化复原。与轻量部署及结构贡献边界直接相关。|
|15|MIE-0015|A / 90|code|[AVC2-UESTC/DRNet-AiO](../resources/MIE-0015/identity.json)|同组动态重参数化官方仓库。核发布源码的可得性与论文结构/条件。|
|16|MIE-0016|A / 90|paper|[Revisiting Lightweight Low-Light Image Enhancement: From a YUV Color Space Perspective](../resources/MIE-0016/identity.json)|同作者YUV轻量低光增强。是结构和效率差异化的重要接续工作，需核是否含时域成分。|
|17|MIE-0017|A / 90|code|[LukasYan30/L3IE-YUV](../resources/MIE-0017/identity.json)|YUV接续实现。核训练/推理、颜色空间和轻量预算的实际代码条件。|
|18|MIE-0018|A / 90|paper|[Towards lightest low-light image enhancement architecture for mobile devices](../resources/MIE-0018/identity.json)|同作者极轻量低光增强架构。优先核查其效率和训练设定，作为移动端预算与差异化的直接约束。|
|19|MIE-0019|A / 90|code|[mubaisam/LiteIE](../resources/MIE-0019/identity.json)|极轻量低光增强实现。核参数统计、迭代步骤和手机速度对应的实现。|
|20|MIE-0020|A / 90|code|[w-xb/MobileIE-6Ch](../resources/MIE-0020/identity.json)|以MobileIE命名的挑战赛六通道变体。核与原4K模型、输入和完整推理量的实质区别。|
|21|MIE-0059|A / 90|paper|[ULightIF: Unsupervised Low Light Infrared and Visible Image Fusion Method](../resources/MIE-0059/identity.json)|可关联固定官方实现具体核查模型、监督和评价；不可访问主文保持独立证据限制。|
|22|MIE-0114|A / 90|code|[cmuhang/ULightIF](../resources/MIE-0114/identity.json)|有固定网络/训练/测试源码与原论文权重入口，能具体核查信息条件和复现路径。|
|23|MIE-0021|A / 80|paper|[FastSHADE: Fast Self-augmented Hierarchical Asymmetric Denoising for Efficient inference on mobile devices](../resources/MIE-0021/identity.json)|移动端高效图像去噪。噪声和完整手机成本是照片到视频扩展的关键约束。|
|24|MIE-0022|A / 80|paper|[JND-Guided Light-Weight Neural Pre-Filter for Perceptual Image Coding](../resources/MIE-0022/identity.json)|JND引导的轻量感知编码预滤波。可为可见闪烁/感知目标提供线索，但图像编码与时序评价不能混同。|
|25|MIE-0023|A / 80|code|[viplab-fudan/FJNDF-Pytorch](../resources/MIE-0023/identity.json)|感知编码预滤波实现。核JND目标、训练数据与实际部署算子，明确与视频感知评价的距离。|
|26|MIE-0024|A / 80|paper|[Bridging the Training-Deployment Gap: Gated Encoding and Multi-Scale Refinement for Efficient Quantization-Aware Image Enhancement](../resources/MIE-0024/identity.json)|量化感知增强与训练部署差距。可核查低精度实现、输入编码与精炼结构的端侧影响。|
|27|MIE-0025|A / 80|code|[GenAI4E/QATIE](../resources/MIE-0025/identity.json)|量化增强实现。核定点/浮点、编码及推理导出路径，而非只看宣称时延。|
|28|MIE-0026|A / 80|paper|[Zero-Shot Illumination and Noise Estimation for Edge-Deployable Low-Light Image Enhancement in IoT Systems](../resources/MIE-0026/identity.json)|面向边缘IoT的零样本照明/噪声估计。训练可得性、噪声假设和部署成本具有直接参考价值。|
|29|MIE-0027|A / 80|paper|[Lightweight unsupervised image enhancement based on bilateral grid](../resources/MIE-0027/identity.json)|双边网格的轻量无监督图像增强。变换表示和计算预算可能影响时序设计，需核实具体信息路径。|
|30|MIE-0028|A / 80|paper|[Energy-Aware Deployment of Quantized Shallow-Water Image Enhancement on an Edge NPU for Underwater IoT Imaging](../resources/MIE-0028/identity.json)|边缘NPU量化水下增强及能耗部署。完整硬件/输入/能量口径可能直接约束效率结论。|
|31|MIE-0029|A / 80|paper|[Efficient Image Denoising on Smartphone GPUs: Mobile AI 2026 Challenge Report](../resources/MIE-0029/identity.json)|Mobile AI 2026手机GPU去噪挑战报告。主要价值为部署、计时与质量评价协议的现实参考。|
|32|MIE-0030|A / 80|paper|[Learned Smartphone ISP on Mobile GPUs: Mobile AI 2026 Challenge Report](../resources/MIE-0030/identity.json)|Mobile AI 2026手机GPU ISP挑战报告。需核硬件、RAW/RGB条件及不同模型的比较范围。|
|33|MIE-0115|A / 80|code|[aiff22/MAI-2025-Workshop](../resources/MIE-0115/identity.json)|三个固定转换脚本可核查实际格式、输入及转换依赖，不等于已在手机成功运行。|
|34|MIE-0031|A / 80|paper|[NTIRE 2026 Challenge on Efficient Low Light Image Enhancement: Methods and Results](../resources/MIE-0031/identity.json)|NTIRE 2026高效低光增强挑战。可核MobileIE参照、变体、预算及端侧质量效率协议。|
|35|MIE-0116|A / 80|paper|[LC-ISP Benchmark: Learning Advanced ISP with Live and Capture Streams](../resources/MIE-0116/identity.json)|主文、公开项目与数据入口可核查流定义、划分、损失和评价约束。|
|36|MIE-0117|A / 80|code|[CVL-UESTC/LC-ISP](../resources/MIE-0117/identity.json)|完整固定源码及配置/数据入口可具体核查预处理、划分和评估。|
|37|MIE-0032|A / 80|paper|[NTIRE 2025 Challenge on Low Light Image Enhancement: Methods and Results](../resources/MIE-0032/identity.json)|NTIRE 2025报告包含MobileIE方案。用于追踪版本、任务和竞赛训练条件，不能直接当视频证据。|
|38|MIE-0033|A / 80|paper|[RGB Photo Enhancement on Mobile GPUs, Mobile AI 2025 Challenge: Report](../resources/MIE-0033/identity.json)|Mobile AI 2025手机GPU照片增强报告。为原始端侧部署、计时条件和真实比较提供线索。|
|39|MIE-0118|A / 80|paper|[Learned Smartphone ISP on Mobile GPUs, Mobile AI 2025 Challenge: Report](../resources/MIE-0118/identity.json)|完整报告可核对任务数据、移动评价、方法和比赛条件。|
|40|MIE-0034|A / 80|code|[gugugu531/Hispark_SocChina_2026](../resources/MIE-0034/identity.json)|HiSilicon相关项目明示MobileIE监督/参考。核算子、模型与端侧成本条件。|
|41|MIE-0035|A / 80|code|[Foadsf/neural-enhance-bench](../resources/MIE-0035/identity.json)|移动增强工具含MobileIE命名的处理类。优先核是否真实加载官方模型及其比较是否公平。|
|42|MIE-0036|A / 80|code|[38260/MobileIE-repro](../resources/MIE-0036/identity.json)|独立复现项目。核其实际复现范围、差异和材料可靠性，不能将说明等同独立实验。|
|43|MIE-0037|A / 70|code|[clarkren/LLVE-AMNet](../resources/MIE-0037/identity.json)|AnyMod代码变体。先辨与官方固定版本的差异，不把fork自动计为独立方法。|
|44|MIE-0038|A / 70|paper|[IGDNet: Zero-Shot Robust Underexposed Image Enhancement via Illumination-Guided and Denoising](../resources/MIE-0038/identity.json)|同作者零样本欠曝增强与去噪。可核查噪声处理、监督方式和图像到视频适配边界。|
|45|MIE-0039|A / 70|code|[LukasYan30/IGDNet](../resources/MIE-0039/identity.json)|零样本增强/去噪官方实现。核优化过程、单图条件和潜在视频适配开销。|
|46|MIE-0040|A / 70|paper|[AI-Driven Illumination-Adaptive Enhancement for Robust Mobile AR Games](../resources/MIE-0040/identity.json)|移动AR游戏中的照明自适应增强。交互场景与稳定性相关，但是否存在跨帧机制需正文核实。|
|47|MIE-0041|A / 70|paper|[Edge-Aware Diffusion for Mobile Photo Enhancement: ASystematic Review and Comparative Latency Analysis](../resources/MIE-0041/identity.json)|明确提及MobileIE的系统综述/时延比较预印本。需优先核引用、模型身份和实测口径，不能把综述说法当独立证实。|
|48|MIE-0119|A / 70|code|[vinodyas/Edge_Aware_Diffusion_for_Mobile_Photo_Enhancement](../resources/MIE-0119/identity.json)|有Android与Python材料可作有限成本/方法一致性核对，未见MobileIE模型复现充分证据。|
|49|MIE-0042|A / 70|code|[Kazuhito00/Multimodal-Node-Editor](../resources/MIE-0042/identity.json)|节点编辑器有MobileIE ONNX入口。核尺寸、前后处理及图像/视频调用条件。|
|50|MIE-0043|A / 70|code|[SKKUAutoLab/ETSS-07-ImageEnhance](../resources/MIE-0043/identity.json)|综合增强框架收录MobileIE代码和配置。核上游版本、封装差异与评价条件。|
|51|MIE-0044|A / 70|code|[khaihoang004/dewarp](../resources/MIE-0044/identity.json)|去畸变项目中的MobileIE命名精炼模块。核原模型忠实度与具体应用路径。|
|52|MIE-0045|A / 70|code|[Barry1749/MobileIE](../resources/MIE-0045/identity.json)|含明显源码/配置差异的fork。大归档未完整取得，固定树文本已另取；实质改动待核。|
|53|MIE-0046|A / 70|other|[TheComplianceAide/Drone_EnhancedVision : MobileIE related research/application material](../resources/MIE-0046/identity.json)|夜视/无人机项目的研究综合资料将MobileIE纳入实时图像/视频路线。核论据与实际实现范围。|
|54|MIE-0047|B / 60|paper|[A Two-Stage Exposure Controllable Network for Image Enhancement With Low-Light Degradation](../resources/MIE-0047/identity.json)|两阶段可控曝光图像增强。曝光控制与帧间亮度稳定存在潜在联系，但时域机制尚未核实。|
|55|MIE-0120|B / 60|code|[cherrysherryplus/TSECN](../resources/MIE-0120/identity.json)|固定训练/测试和权重资料可用于有限设计/评估边界核查。|
|56|MIE-0048|B / 60|paper|[Real-time minimalist imaging via spatial aberration uniformization and re-parameterized restoration](../resources/MIE-0048/identity.json)|空间像差均匀化与重参数化复原。可核查训练/部署转换与成像任务差异。|
|57|MIE-0049|B / 60|paper|[Decoupling color and structure via dual-branch learning for high-fidelity underwater image restoration](../resources/MIE-0049/identity.json)|颜色/结构双分支水下图像复原。与表示分工、颜色稳定性相关，但时域效果尚待核验。|
|58|MIE-0121|B / 60|code|[Qlinlin/UDHE](../resources/MIE-0121/identity.json)|可定位作者、发布承诺和未来复现条件；当前只有README占位，技术内容须依论文，E单独标记。|
|59|MIE-0050|B / 60|other|[heniath/underwater-image-enhancement : MobileIE related research/application material](../resources/MIE-0050/identity.json)|水下增强项目中的文献/预算比较资料。可核设备、分辨率与来源口径，不把汇编表当同轮实测。|
|60|MIE-0051|B / 50|paper|[StereoAdapter: Adapting Stereo Depth Estimation to Underwater Scenes](../resources/MIE-0051/identity.json)|水下场景立体深度适配。与增强任务不同，可用于核查增强作为前处理的下游联系及信息条件。|
|61|MIE-0052|B / 50|code|[AIGeeksGroup/StereoAdapter](../resources/MIE-0052/identity.json)|水下立体适配作者代码。核增强前处理关系和信息条件，非直接时域增强结论。|
|62|MIE-0053|B / 50|paper|[Detection-Gated Adaptive Region-of-Interest Reconstruction for single-lens infrared computational imaging](../resources/MIE-0053/identity.json)|单镜头红外计算成像的检测门控ROI重建。局部计算与门控预算有参考线索，但不是现成的视频增强对照。|
|63|MIE-0054|B / 50|paper|[Beyond Pipeline Mimicry: Expert-level Aesthetic ISP via Reward-Guided Flow](../resources/MIE-0054/identity.json)|奖励引导flow的美学ISP。需区分生成/概率flow与运动光流，核查质量目标与部署条件。|
|64|MIE-0055|B / 50|code|[horizon921/Computational-Photography-26Fall-MobileIE](../resources/MIE-0055/identity.json)|同主题课程项目的不同快照。需逐项比较，不能按名称预设与其他复现完全相同。|
|65|MIE-0056|B / 50|code|[wenqi-deng/Computational-Photography-26Fall-MobileIE](../resources/MIE-0056/identity.json)|计算摄影课程相关复现/变体。核其独立改动、训练条件和原MobileIE版本关系。|
|66|MIE-0057|B / 50|code|[miaory/test](../resources/MIE-0057/identity.json)|ONNX导出文件中出现MobileIE类/路径。核有效代码与注释/样例的区别。|
|67|MIE-0058|B / 50|other|[谢袁学习主页：MobileIE 与小模型蒸馏阅读资料](../resources/MIE-0058/identity.json)|公开阅读笔记将MobileIE与蒸馏/训练部署联系。核笔记实际支持范围，不当作已有实验。|
|68|MIE-0060|B / 40|paper|[Frequency-Domain Optimized Laplacian Pyramid Network for Weld DR Image Enhancement](../resources/MIE-0060/identity.json)|焊缝DR图像的频域/金字塔增强。可作空间增强先例，不先验认为涉及自然视频或移动端部署。|
|69|MIE-0061|B / 40|paper|[CS³-Diff: Collaborative spatio-spectral-scale guided diffusion for low-light image enhancement](../resources/MIE-0061/identity.json)|协同多域/多尺度扩散低光增强。主要核查与轻量因果视频目标的条件及成本差异。|
|70|MIE-0122|B / 40|code|[ivsou/CS--Diff](../resources/MIE-0122/identity.json)|主要核引用/实现关系、空间与扩散步术语及计算边界，暂未见直接轻量视频迁移用途。|
|71|MIE-0062|B / 40|paper|[EPIDIR: Integrating external frequency prompts and internal statistical decoupling for infrared image restoration](../resources/MIE-0062/identity.json)|外部频率提示与内部统计解耦的红外复原。侧重核实其MobileIE引用，避免把任务迁移当视频机制。|
|72|MIE-0063|B / 40|paper|[MorphSIG: Subject-Driven Image Generation via Decoupled Anchoring and Feature Transport](../resources/MIE-0063/identity.json)|同作者主题驱动图像生成与特征传输。标题中的传输不自动代表真实视频一致性；先查引用与机制边界。|
|73|MIE-0064|B / 40|code|[LukasYan30/MorphSIG](../resources/MIE-0064/identity.json)|主题生成作者代码。核所谓特征传输是否为视频机制，避免名称相近造成误判。|
|74|MIE-0065|B / 40|code|[LukasYan30/MobileIE](../resources/MIE-0065/identity.json)|原作者个人fork的不同树。主要核版本历史，而非预设新方法。|
|75|MIE-0066|B / 40|code|[songyang86/MobileIE](../resources/MIE-0066/identity.json)|不同树的MobileIE fork。核其是否仅早期版本/文档变化。|
|76|MIE-0067|B / 40|code|[strategist922/MobileIE](../resources/MIE-0067/identity.json)|不同树的MobileIE fork。辨别版本差异和独立技术内容。|
|77|MIE-0068|B / 40|code|[AVC2-UESTC/NTIRE2025-LLIE](../resources/MIE-0068/identity.json)|官方挑战仓库。需核是否实际发布方法源码，不能以官方仓库存在代替代码可得。|
|78|MIE-0069|B / 40|other|[MobileIE 中英文解读、转载与传播资料族](../resources/MIE-0069/identity.json)|中英文解读/转载族。主要核来源链与效率口径误传，不把多次转述当独立证据。|
|79|MIE-0070|B / 40|other|[Synapse-iiserk/beautiful_ppt : MobileIE related research/application material](../resources/MIE-0070/identity.json)|水下AI演示材料涉及MobileIE。主要用于核对解释与技术来源，不预设有独立方法贡献。|
|80|MIE-0071|B / 40|other|[38260/topconf-4-pillars : MobileIE related research/application material](../resources/MIE-0071/identity.json)|结构化论文分析/核验项目收录MobileIE。核原文引用和自动生成解释的证据可靠性。|
|81|MIE-0072|B / 40|other|[dnguyennk/portfolio : MobileIE related research/application material](../resources/MIE-0072/identity.json)|公开项目展示涉及MobileIE。核展示指向、实现可得性及可审查结果。|
|82|MIE-0123|B / 40|other|[Chris23333/mobile-ai-image-denoising-papers : MobileIE related third-party literature digest](../resources/MIE-0123/identity.json)|主要用于来源校正与候选线索，非实际模型/设备证据。|
|83|MIE-0124|C / 30|paper|[Target-Focused Perception for Embodied Question Answering](../resources/MIE-0124/identity.json)|主要用于引用语境与下游联系核查，完整方法尚无可得原文支持。|
|84|MIE-0073|C / 30|paper|[CAS-HDPA-Yolov11: a dual attention enhanced network for monocular human height estimation](../resources/MIE-0073/identity.json)|单目身高估计检测中的HDPA。说明基础模块可能被复用，实际增益与二期任务不等同。|
|85|MIE-0074|C / 30|paper|[Spatial-Aware Modulation Hierarchical Attention for Visible and Infrared Image Fusion](../resources/MIE-0074/identity.json)|可见/红外融合的层次注意力。重点核实HDPA等组件来源，不直接当视频增强新机制。|
|86|MIE-0075|C / 30|code|[369pro/DEIMv8](../resources/MIE-0075/identity.json)|369pro/DEIMv8 的公开命中涉及MobileIE的MBRConv/HDPA/模型定义或配置。逐项核查具体复用/抄录与实现条件，不将它当独立视频方法。|
|87|MIE-0076|C / 30|code|[Alleluia1/RTDETR-10.20](../resources/MIE-0076/identity.json)|Alleluia1/RTDETR-10.20 的公开命中涉及MobileIE的MBRConv/HDPA/模型定义或配置。逐项核查具体复用/抄录与实现条件，不将它当独立视频方法。|
|88|MIE-0077|C / 30|code|[Alleluia1/RTDETR-main-distill1](../resources/MIE-0077/identity.json)|Alleluia1/RTDETR-main-distill1 的公开命中涉及MobileIE的MBRConv/HDPA/模型定义或配置。逐项核查具体复用/抄录与实现条件，不将它当独立视频方法。|
|89|MIE-0078|C / 30|code|[FangruLinghu/PaperCode](../resources/MIE-0078/identity.json)|FangruLinghu/PaperCode 的公开命中涉及MobileIE的MBRConv/HDPA/模型定义或配置。逐项核查具体复用/抄录与实现条件，不将它当独立视频方法。|
|90|MIE-0079|C / 30|code|[FifthGentleness/RTDETRU](../resources/MIE-0079/identity.json)|FifthGentleness/RTDETRU 的公开命中涉及MobileIE的MBRConv/HDPA/模型定义或配置。逐项核查具体复用/抄录与实现条件，不将它当独立视频方法。|
|91|MIE-0080|C / 30|code|[Wzhi997/YOLO](../resources/MIE-0080/identity.json)|Wzhi997/YOLO 的公开命中涉及MobileIE的MBRConv/HDPA/模型定义或配置。逐项核查具体复用/抄录与实现条件，不将它当独立视频方法。|
|92|MIE-0081|C / 30|code|[XuYuanDaWang/Yolo-Project](../resources/MIE-0081/identity.json)|XuYuanDaWang/Yolo-Project 的公开命中涉及MobileIE的MBRConv/HDPA/模型定义或配置。逐项核查具体复用/抄录与实现条件，不将它当独立视频方法。|
|93|MIE-0082|C / 30|code|[benderick/DEIM-DEIM](../resources/MIE-0082/identity.json)|benderick/DEIM-DEIM 的公开命中涉及MobileIE的MBRConv/HDPA/模型定义或配置。逐项核查具体复用/抄录与实现条件，不将它当独立视频方法。|
|94|MIE-0083|C / 30|code|[ccz308476-ai/General_yu](../resources/MIE-0083/identity.json)|ccz308476-ai/General_yu 的公开命中涉及MobileIE的MBRConv/HDPA/模型定义或配置。逐项核查具体复用/抄录与实现条件，不将它当独立视频方法。|
|95|MIE-0084|C / 30|code|[cui123000/rtdetr_indoor](../resources/MIE-0084/identity.json)|cui123000/rtdetr_indoor 的公开命中涉及MobileIE的MBRConv/HDPA/模型定义或配置。逐项核查具体复用/抄录与实现条件，不将它当独立视频方法。|
|96|MIE-0085|C / 30|code|[daidaidaidaidaidai-cat/duomotai-yolo](../resources/MIE-0085/identity.json)|daidaidaidaidaidai-cat/duomotai-yolo 的公开命中涉及MobileIE的MBRConv/HDPA/模型定义或配置。逐项核查具体复用/抄录与实现条件，不将它当独立视频方法。|
|97|MIE-0086|C / 30|code|[gogoworker/FSOD](../resources/MIE-0086/identity.json)|gogoworker/FSOD 的公开命中涉及MobileIE的MBRConv/HDPA/模型定义或配置。逐项核查具体复用/抄录与实现条件，不将它当独立视频方法。|
|98|MIE-0087|C / 30|code|[helloshuaige520/RT-DETR](../resources/MIE-0087/identity.json)|helloshuaige520/RT-DETR 的公开命中涉及MobileIE的MBRConv/HDPA/模型定义或配置。逐项核查具体复用/抄录与实现条件，不将它当独立视频方法。|
|99|MIE-0088|C / 30|code|[lightxiu/yolo26](../resources/MIE-0088/identity.json)|lightxiu/yolo26 的公开命中涉及MobileIE的MBRConv/HDPA/模型定义或配置。逐项核查具体复用/抄录与实现条件，不将它当独立视频方法。|
|100|MIE-0089|C / 30|code|[linchuan232/myrtdetr](../resources/MIE-0089/identity.json)|linchuan232/myrtdetr 的公开命中涉及MobileIE的MBRConv/HDPA/模型定义或配置。逐项核查具体复用/抄录与实现条件，不将它当独立视频方法。|
|101|MIE-0090|C / 30|code|[lrbob/deim](../resources/MIE-0090/identity.json)|lrbob/deim 的公开命中涉及MobileIE的MBRConv/HDPA/模型定义或配置。逐项核查具体复用/抄录与实现条件，不将它当独立视频方法。|
|102|MIE-0091|C / 30|code|[mahongwei-bot/RTDETR](../resources/MIE-0091/identity.json)|mahongwei-bot/RTDETR 的公开命中涉及MobileIE的MBRConv/HDPA/模型定义或配置。逐项核查具体复用/抄录与实现条件，不将它当独立视频方法。|
|103|MIE-0092|C / 30|code|[mahongwei-bot/RTDETR20260307-main](../resources/MIE-0092/identity.json)|mahongwei-bot/RTDETR20260307-main 的公开命中涉及MobileIE的MBRConv/HDPA/模型定义或配置。逐项核查具体复用/抄录与实现条件，不将它当独立视频方法。|
|104|MIE-0093|C / 30|code|[pbobip/wd_bishe](../resources/MIE-0093/identity.json)|pbobip/wd_bishe 的公开命中涉及MobileIE的MBRConv/HDPA/模型定义或配置。逐项核查具体复用/抄录与实现条件，不将它当独立视频方法。|
|105|MIE-0094|C / 30|code|[pls-Kaguya/LSO-RT-DETR](../resources/MIE-0094/identity.json)|pls-Kaguya/LSO-RT-DETR 的公开命中涉及MobileIE的MBRConv/HDPA/模型定义或配置。逐项核查具体复用/抄录与实现条件，不将它当独立视频方法。|
|106|MIE-0095|C / 30|code|[tpiece/RTDETR-main](../resources/MIE-0095/identity.json)|tpiece/RTDETR-main 的公开命中涉及MobileIE的MBRConv/HDPA/模型定义或配置。逐项核查具体复用/抄录与实现条件，不将它当独立视频方法。|
|107|MIE-0096|C / 30|code|[wbzuo/dl-modules-toolkit](../resources/MIE-0096/identity.json)|wbzuo/dl-modules-toolkit 的公开命中涉及MobileIE的MBRConv/HDPA/模型定义或配置。逐项核查具体复用/抄录与实现条件，不将它当独立视频方法。|
|108|MIE-0097|C / 30|code|[wujunye1212/wjy_vault](../resources/MIE-0097/identity.json)|wujunye1212/wjy_vault 的公开命中涉及MobileIE的MBRConv/HDPA/模型定义或配置。逐项核查具体复用/抄录与实现条件，不将它当独立视频方法。|
|109|MIE-0098|C / 30|code|[xuhangc/Plumbing](../resources/MIE-0098/identity.json)|xuhangc/Plumbing 的公开命中涉及MobileIE的MBRConv/HDPA/模型定义或配置。逐项核查具体复用/抄录与实现条件，不将它当独立视频方法。|
|110|MIE-0099|C / 30|other|[objectdetection_script：MobileIE MBRConv3移植至RT-DETR的公开变更说明](../resources/MIE-0099/identity.json)|RT-DETR公开变更说明声称移植MBRConv3。核公开说明、可得代码与商业材料边界。|
|111|MIE-0100|C / 20|paper|[Multi-Type Ship Detection in Complex Marine Backgrounds Using an Enhanced YOLO-Based Network](../resources/MIE-0100/identity.json)|海洋场景多类型舰船检测。主要价值在MobileIE组件或基线的实际引用边界。|
|112|MIE-0101|C / 20|paper|[YOLOv11-SMS: An Improved Algorithm for Impurity Detection in Seed Cotton](../resources/MIE-0101/identity.json)|籽棉杂质检测。保留直接引用，主要检查轻量/注意力组件的引用来源和任务边界。|
|113|MIE-0102|C / 20|paper|[MAEL-DET: a multi-scale adaptive enhanced lightweight network for a welding defect detection algorithm](../resources/MIE-0102/identity.json)|焊接缺陷检测。轻量网络模块来源与增强任务不同，方法关联需要正文证据。|
|114|MIE-0103|C / 20|paper|[A Multi-Modal Fusion Network of Visible–Ultraviolet Features for Detecting Aero-Engine Blade Microcracks](../resources/MIE-0103/identity.json)|可见/紫外融合微裂纹检测。多模态缺陷任务需与低光RGB视频的信息条件分开。|
|115|MIE-0104|C / 20|paper|[ViMAF-YOLO: a lightweight detection model integrating multi-scale feature fusion and attention mechanisms](../resources/MIE-0104/identity.json)|轻量检测网络中的多尺度融合/注意力。先核实MobileIE引用语境，不预设能迁移为视频增强机制。|
|116|MIE-0105|C / 20|paper|[VHSS-Det: Bridging Scale Discrepancies and Morphological Distortions for View-Invariant Heterogeneous Security Sensing](../resources/MIE-0105/identity.json)|异构安全感知目标检测。需核实引用与尺度/形变建模的具体关系，暂未见直接增强用途。|
|117|MIE-0106|C / 20|paper|[Research on Lightweight PCB Defect Detection Algorithm based on YoloV11](../resources/MIE-0106/identity.json)|轻量PCB缺陷检测。属于异任务引用，分析其组件复用而不将其计作视频增强先例。|
|118|MIE-0107|C / 20|paper|[Robust object detection in optical remote sensing imagery via partial-channel self-attention and intensity-enhanced feature fusion](../resources/MIE-0107/identity.json)|光学遥感目标检测与部分通道注意力。属于跨任务模块引用，不能凭标题认定增强算法贡献。|
|119|MIE-0108|C / 20|paper|[Escalator Falling Behavior Detection Algorithm Based on YOLOv5-MSPP](../resources/MIE-0108/identity.json)|扶梯跌倒行为检测。视频应用标题不自动代表增强或闪烁处理，重点核实引用的实际用途。|
|120|MIE-0109|C / 10|paper|[HDS ‐Net: A Hybrid Deformable‐Swin Network for Dual‐Domain Sparse‐View CT Reconstruction](../resources/MIE-0109/identity.json)|双域稀疏视角CT复原。与手机RGB视频差异大，主要核查为何引用MobileIE。|
|121|MIE-0110|C / 10|paper|[A compact and interpretable multi-source framework for heterogeneous medical image classification](../resources/MIE-0110/identity.json)|多源医学图像分类框架。任务远离视频增强，仍逐项核查引用和模型规模比较是否准确。|
|122|MIE-0111|C / 10|code|[W-ayivi/ML-ConvNet](../resources/MIE-0111/identity.json)|既有被引医学论文原文明确给出的作者代码。核引用/规模比较语境，不预设复用了MobileIE模型。|
|123|MIE-0112|C / 10|paper|[3-D Ultrasonic Imaging Method for Composite Materials Based on Fusion of Visible Light Image and Depth Maps](../resources/MIE-0112/identity.json)|复合材料三维超声成像与图像/深度融合。信息传感条件离题较远，仍核查引用范围。|
|124|MIE-0113|C / 10|paper|[Co-umamba: Collaborative global–local modeling for precise apple leaf disease segmentation](../resources/MIE-0113/identity.json)|苹果叶病分割的全局局部建模。主要核查轻量模块引用，不把分割收益当增强收益。|
