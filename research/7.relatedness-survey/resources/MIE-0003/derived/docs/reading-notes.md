# MIE-0003｜AnyMod-LLVE阅读与代码核对

2026-11-20，Codex  (sqs_A100)；快照截止2026-11-17。MIE-0002关闭后开始本项。阅读全文17页，包括正文、参考文献和同PDF中的附录；并核作者官方实现的论文相关路径。未开始MIE-0004独立项目分析，未执行任何模型。

## 身份与证据版本

Hangfeng Liang、Yutao Hu、Yanhan Hu、Xiaohan Wu、Wenqi Shao、Ying Fu；arXiv:2606.11186v1。abs元数据提交日2026-06-09；PDF页脚写Preprint June10,2026，分别记录，不能当会议录用/发表日期。索引初卡的Liang Hangfeng语序按原PDF作者改为Hangfeng Liang。arXiv DOI是预印本标识，不据此声称已正式会议发表。

原件SHA256 2c57ce78f51438ecdf9e1af0b69275d3f92296771e35d71470f6a78e057b3976；原始获取含一个HTTP202空体，不能误当全文，实际全文来自成功的arXiv PDF。可解析原件、逐页图和全文提取见[paper-assets.json](paper-assets.json)，由私有缓存链接保存，不在Git再发布全文。官方论文代码链接与README对应lhfgghc/LLVE-AMNet，固定9dcbe4db5e2a0710cc020936a80b32df8ca712a6，提交2026-08-08晚于预印本，故代码差异不自动说明发表时实验如何运行。

实际查看17页接触表，放大查看p4、p6、p7、p8、p12、p13、p14、p15、p16、p17。图1范式示意可由接触表辨识；主要架构、数值表、补充图表均单页查看。静态裁图不称观看过视频或证明无闪烁。

## 分节逐页阅读

| 页／节 | 内容、联系与限制 |
|---|---|
| p1 摘要、§1、Fig1 | 低光视频；事件/红外辅助在测试可能缺失，生成隐式特征补充。主问题是模态可用性，不是MobileIE的4K参数手机约束。 |
| p2 §1、§2.1 | S2DG译码器、空间可靠性与频域选择；以v2e/ThermalGen生成预训练辅助。大规模训练先验不等于RGB推理时获得新的真实传感器观测。 |
| p3 §2、§3.1 | 回顾时间优化与缺失模态学习；输入RGB视频，每时刻可选事件/IR。真实事件变体须另核采集时间窗/同步，不能只由RGB路径因果就认定全部模态条件在线。 |
| p4 §3.1–3.2、Figs2–3 | RGB多尺度编码、可选真实或生成辅助、融合、ConvLSTM(H/C状态)、解码残差；IADS低频/高频分解。架构图确有跨时刻状态箭头，和单帧MobileIE不同。 |
| p5 §3.2–3.3、Eqs1–12 | 空间sigmoid门、FFT频谱门与tanh缩放、iFFT、低频残差；重建L1+SSIM、所有模态组合监督、normalized feature L2及stop-grad教师。没有显式运动对齐/遮挡机制或时间损失公式。 |
| p6 §4.1–4.2、Table1、Fig4 | 预训练合成曝光10–50%及1–10%，4×A800；下游声称1×A800/batch32。clip8、crop128²、AdamW2e-4/cosine/warm5。DID413视频/41038帧及8:2划分；SDSD室内外。SDE描述43室外48室内、每视频145888帧，须保留数据口径疑点。RGB-only指微调/推理输入，预训练仍有多模态生成监督。Table1报告DID31.57/.95等空间指标，无时间/端侧成本指标。 |
| p7 §4.2、Tables2–3、Fig5 | Table2 SDE真实event、额外IR来自ThermalGen（非真实IR采集）。Table3明确MobileIE零样本16.03/.74、16.42/.71、16.10/.48，作者方法25.07/.93、22.27/.87、21.43/.74。仅作者报告，未给MobileIE具体权重、分辨率/裁剪/颜色/预算匹配流程。Fig5静态细节、模态缺失比较不能代替时间一致性评价。 |
| p8 §4.2–5、Tables4–5、Fig6 | Table4预训练规模与距离；Table5 IADS/FBS消融。FBS-only SDSD-indoor29.20高于组合29.03，故“组合各数据集最好”与表不全符。Table4所有feature L2随规模单调下降也不准确：SDSD-indoor IR .208→.211，outdoor event .206→.210→.203。Fig6是热图示例，非统计分布拟合或时间稳定实证。 |
| p9–11 | 结论、致谢、impact和完整参考文献。p11准确引用MobileIE arXiv2507.01838；p8还把MobileIE归入restoration foundation model比较组，这是本文分组，不能改变MobileIE原任务和训练规模事实。后向参考不扩展成全部视频领域收集。 |
| p12 附录A/B、Table6 | RetinexFormer骨干，通道[64,128,256,256]、H/8 latent256、SNR融合借鉴EvLight++、ConvLSTM状态、100轮/EMA等。没有给手机时延、峰值RAM、能量或完整模型算量表。 |
| p13 附录C/D、Table7 | Vimeo90K、REDS、YouTube-VOS、Inter4K、UVO、MOSEv2，合102112视频、1693034帧；表格求和吻合。不同源的时间采样步长分别描述；尚未验证原视频/帧的独立性、许可或与下游测试内容重叠。 |
| p14 附录D/E/F、Fig7 | 空间/频谱门图，额外zero-shot和missing-modality说明。热图只能解释示例激活，不是因果消融或无幻觉证明。 |
| p15 附录F、Fig8 | 所谓feature distribution comparison实际为真实/生成特征热图，无散点分布/统计检验；本文文字“overlapping distributions”不能据图升级为校准良好。 |
| p16 Figs9–10 | 不同模态静态增强、zero-shot静态裁图明确出现MobileIE。未查看动态视频，本页不证明运动场景下稳定或MobileIE闪烁。 |
| p17 Table8 | RGB-only inference Full Modality training：DID30.35，与Table1/4的31.57不同；条件差别未解释。B组outdoor R+E26.35大于R+E+I26.29，正文“全模态最好”不适用该PSNR列。分别留数值，不混成同一结果。 |

## 官方源码的论文相关核查

[源码清单](upstream-code-manifest.json)保留21个文件和SHA：README、6个核心model文件、3个配置、trainer、metrics、train入口、DID/SDE-MM/SDSD数据类、5个split JSON。实际逐行读取适用Python/YAML/README；JSON只作数据列表解析/比较，不声称逐个视频内容阅读。未执行外部代码，未安装依赖，未取/加载云盘权重；纸面数值始终为作者报告。

- **机制**：adapter.py从四层特征投影到H/8、AvgPool3×3分低高频，实虚通道拼接做rfft2门和缩放、irfft2、额外ctx_guide和reconstruct_head；比论文简式更具体。amnet.py单向按t循环ConvLSTM；RGB-only路径未发现读取t之后帧。离线inference_video每次清零H/C；start_stream/stream_step/end_stream提供跨调用状态，但stream_step本身不调用eval，调用者需设评估模式。首帧状态零，无自动切镜头检测；size变化需重新start。
- **训练目标**：源码pixel是Charbonnier(eps1e-3)而非逐字L1；normalized教师蒸馏为1−cos，若向量单位化则与平方L2相差常数2，不能简单断言不同方向，但权重和平均方式不能照抄公式。源码按combos平均重建，loss weights/蒸馏warm-up在配置。utils.metrics里有temporal_loss函数，当前amnet训练目标和trainer路径未调用它，不能据函数存在宣称用时间损失训练。
- **预训练公平性**：README声明所有微调共享amnet_pretrained.pth；公开此ZIP主要为下游训练入口，未见表7数据构建/v2e/ThermalGen完整流水线或预训练配置。需单独比较同数据/同训练预算与多模态预训练收益，不能仅相同推理RGB就称所有训练条件公平。
- **参数与内存算术**：默认ConvLSTMCell(256→256,k3)参数=(256+256)×(4×256)×9+4×256=4,719,616，仅这一层已远高于MobileIE整网约4K；不是AMNet整网总参数或实测。1080p、H/8×W/8、B1、1层、H/C两张256通道FP32状态为66,355,200B=63.28125MiB，不含编码多尺度、FFT/attention、输出/辅助记录、工作区。FP16纯状态算术可减半，但不证明完整运行时如此。标准卷积融合不能消去跨帧状态、FFT或动态attention。
- **默认验证重要缺口**：scripts/train.py test_loader batch_size=4，而trainer.evaluate只使用x[0]/y[0]，每batch其他样本不进入指标；DID/SDSD/SDE类在test也先base_size256再128²中心crop，clip长度8、窗口起点stride2且窗口内逐帧。模型每clip重置，重叠帧可重复计权，不能把这当全分辨率唯一帧连续视频测试。源码事实不等于已证明论文表格实际用该有缺口的路径；作者可能有其他未发布评测脚本。
- **指标**：utils.calculate_psnr统一RGB像素MSE，SSIM由pytorch_msssim，逐帧平均；与MobileIE公开默认逐通道PSNR不同。无显式运动补偿时间指标、闪烁或设备持续成本输出。所有方法需统一到同协议后重比，不能直接拼表。
- **数据**：五split文件字面train/test ID无交集，见[split-static-check.json](split-static-check.json)，DID330/83、SDE文件标indoor36/7、outdoor40/8，SDSD64/6及70/10。SDE两子集总数43/48与p6室内外描述顺序不一致；未取实际场景内容，不能确认哪一方标签错。没有视频内容hash，ID无交集也不证明预训练与测试或同场景无重叠。
- **事件时间/编码**：SDE loader按同名npz读取，不校验文件中事件时间截止是否早于目标帧；取事件时间归一化后floor到单bin，与正文“two closest voxels”不同。IR loader在该配置始终占位缺失，SDE默认use_ir=false，不能从该配置重建Table2的所有IR条件。生成IR推理成本在“真实输入是合成IR”的条件需另计，不能混入RGB-only时延。
- **版本／许可**：README宣称MIT并链接LICENSE，此固定47文件ZIP没有LICENSE；仅报告声明，许可具体授权范围未确认。公开代码晚于论文v1，固定版本与论文表格精确对应仍未知。

## 可证伪研究含义与缺口

已有“较重视频模型、多模态训练、RGB推理、隐式辅助、单向ConvLSTM”组合先例，二期不能把这些宽泛组合直接当首创。其训练期丰富信息和潜在教师价值值得保留，但整个256通道频域/时序模块不能不计成本地塞入MobileIE。

需区分三类反证：同权重清零/打乱/真实历史以确认时间贡献；同RGB视频/预训练预算和空间质量以隔离训练特权；离线clip与真stream、运动/遮挡/切镜头/曝光分别测以反证“更低帧差就是更好”。特征蒸馏到极小学生、低分辨率统计或状态压缩只能作为后续假设，当前无收益证据、新颖性仍待全部调查。

缺口：MobileIE比较权重/代码与完整zero-shot协议；训练预处理和预训练脚本；原数据/许可/场景独立性；表内冲突的作者解释；权重运行及真正流式质量；手机精度/FFT支持/时延/RAM/功耗。所有未知均限制结论，不用作者空间结果证明时序、移动效率或MobileIE失效。0模型执行/训练/GPU实验，未进入Gate3。
