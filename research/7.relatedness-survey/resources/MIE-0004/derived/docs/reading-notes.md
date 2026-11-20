# MIE-0004｜AMNet官方项目静态阅读

2026-11-20；Codex  (sqs_A100)。固定9dcbe4db5e2a0710cc020936a80b32df8ca712a6，47文件。Gate1增补与MIE-0003正式版复审完成后恢复本项，沿用原begin，不重开或跳过下一项。

## 范围与来源

40个可读文本入口及7个原始媒体均已归档。21个论文相关模型/训练/配置/划分文件此前在MIE-0003实际阅读，本项核同一hash并重读关键控制流；其他适用工具、配置、数据、入口和网页已读。HTML仅审发布内容和适用媒体同步逻辑，不称整个CSS/仓库每行审计。逐文件范围见[reading-scope](reading-scope.json)。

README称官方实现并链接arxiv、主页、海报、预训练及五组微调权重。网页/海报有ICML2026标识，正式PMLR已在MIE-0003追加复审确认，不再只依据项目自称。README仍引用arxiv，并标MIT且链接LICENSE；47文件包实际无LICENSE，另无requirements.txt/setup.py/pyproject.toml。安装依赖为README手工pip命令，不是冻结环境。

## 核心实现与状态

- `models/amnet.py`1–118构造Retinex编码/解码、两个S2DG、event/IR编码、SNR融合及ConvLSTM，未见MobileIE模型继承。此为比较MobileIE的论文项目，而非其轻量骨干改造。
- `_forward_impl`149–284在每次调用开始初始化H/C，按t处理单帧；RGB数学依赖为当前/历史帧。整个offline接口接收并返回clip，不等于端到端实时系统；真实event每文件的时间边界未在本项核实。
- `inference_video`383–393会eval并调用clip路径；`start_stream/stream_step/end_stream`395–479可跨调用保留状态，有图像尺寸断言，没有自动切镜头/尺寸重置。stream_step有no_grad但不自动eval，调用方须设置模式。
- 真实模态开关True/False/None分别影响真实/合成特征选择；None有真实输入时选真实latent，但SNR illumination是否接入真实模态由显式True控制。offline与stream均有此规则，未把它误判成两条实现不同的实验证据。
- 源码的`temporal_loss`只在utils/metrics.py定义，未接入默认训练；时间机制来自递归状态等路径，不能据此宣称完全没有时间学习。无跨模型MobileIE基线评测接线或完整相机/编解码stream runner，NPU链接留待MIE-0005单独分析。

## 配置、加载与评价

五个YAML、五个shell入口、四个dataset类及全部split列表已检查。clip8/stride2、短边base256后crop128²、batch4及微调小学习率等见[配置审计](static-config-audit.json)；不沿用论文Table6的batch32/lr2e-4去描述这份发布配置。SDE室内外数量与论文文字的对应差别保留于MIE-0003，未核实际下载视频内容。

`scripts/train.py`141–159只冻结encoder/decoder/ln_img/proj_img，Trainer可另冻辅助编码器；ConvLSTM/translator等仍可能训练，配置注释“only head”不能代替真实参数选择。工具param_count.py只数requires_grad，主train脚本则同时打印trainable/total；未调用任何模型计数或构造模型。

预训练加载有一个可定位的条件路径：无resume且指定文件不存在时，Trainer222–227只警告返回，外层163–167仍打印“Weights loaded”。结合默认冻结模块，这条路径会保留构造初始化而非得到预训练；这是静态条件分析，不说明作者运行过该路径。已有权重则strict=False加载并只打印missing/unexpected数量，真实键/形状/版本是否匹配需另核。

`scripts/train.py`127–134设置测试batch4，而Trainer445–455只取y[0]/x[0]；当前公开路径遗漏同batch其余样本。重叠clip和每clip清状态亦不同于唯一帧计权的连续stream评估。PSNR用全RGB MSE（utils/metrics.py6–15），不同于原MobileIE默认逐通道口径；SSIM/Charbonnier/归一化蒸馏和论文差别仍参照MIE-0003，不改写为作者实际会议脚本。

README明确五配置均从同一预训练模型微调；固定47文件包未给六数据源合成辅助预训练的完整生成/配置路径。Trainer69附近只识别已初始化进程组，给定train.py/launcher没有初始化、DDP包装或分布式sampler调用；这不证明作者未进行四卡预训练，只说明不能据此发布入口声称已经具备其完整复现接线。

调度器以optimizer更新步计warmup/cosine；checkpoint记录model/optimizer/epoch/global_step及可选scheduler/EMA，未把它当完整数据/RNG/设备复现保证。Logger方法为空，但Trainer240–254另有metrics_log.jsonl写入，不能说整个项目没有日志。

## 展示与权重入口

实际查看框架PNG、海报缩略总览及五段视频各6帧接触表。两个建筑立面示例的四个输入/输出文件均405帧、1920×1080、30fps；宣传片2880帧。解码完整源帧用于资产清点，但人工只看抽样，未观看全部视频或测闪烁/MOS。抽样显示作者结果更明亮；无GT/重跑依据，不作真实性或时间稳定胜出判断。

网页`docs/index.html`1181–1200用playbackRate0.94/1.06或0.88/1.12与seek纠正两个播放器漂移，拖动时暂停后重对齐。这是展示层同步，不是AMNet时间算法或感知评测；浏览器呈现不能替代按同帧/同时间的模型比较。页脚“Full release coming soon”与已有源码并存，仅记录，不猜缺失哪些未发布实验。

README七个云盘入口已GET并留原始回执：Google401；六百度页为提取码界面，动态hash不同，未取得二进制权重。公开提取码在原README中；本项不加载/执行权重，不把200页当权重已验证。详见[获取记录](weight-entry-review.json)。

## 二期价值和限制

可移植的是明确区分offline清状态与stream持久状态、可选辅助条件、训练冻结/预训练和评价协议。不能直接移植256通道ConvLSTM/FFT后仍称MobileIE约4K/同手机成本，也不能把展示抽样当消除闪烁的证据。

后续需用同权重同输入比较clip/连续stream/清零与错序状态，统一预处理/指标/帧计权并验证完整加载，再单列辅助预训练和真实模态条件、首帧/遮挡/曝光/切镜头以及完整端侧成本。当前0模型执行、0训练、0GPU、未进入Gate3；权重、实际完整协议、数据内容与授权等缺口均保留。
