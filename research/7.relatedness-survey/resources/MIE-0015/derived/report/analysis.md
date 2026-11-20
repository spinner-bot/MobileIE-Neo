# MIE-0015｜DRNet作者实现：可审查的配置机制与复现入口缺口

2026-11-21；Codex  (sqs_A100) <langxibielangle@qq.com>。范围为固定公开21文件与完整权重载体的适用静态分析，0上游执行/模型加载/训练/GPU/Gate3。

此项目与MIE-0014论文使用同一代码载体，能够支持对任务先验如何折成双bank权重的具体核查。它没有提供已验证的连续视频机制或手机完整成本；默认入口、任务映射、训练预算和评价材料也不能直接覆盖论文全部实验。相同源内容与权重不构成第二次独立数值证实。

[身份卡](../../identity.json)、[来源清单](../../sources/manifest.json)、[21件清点](../docs/archive-inventory.json)、[逐blob/SHA对账](../docs/source-correspondence-audit.json)及[阅读笔记](../docs/reading-notes.md)保存版本与范围。固定提交f1d25d2acc2db0f16e1b8fc0b687bb9eaf7a1e4d、tree d914357cd3e2f25cdfff65e6815bdef370c80436，21件152472B；Git提交时间2026-06-07不是首公开证明。README的8作者和论文官方链接支持作者项目身份；owner本身不是充分依据。

模型核心已在14阅读，全部21件摘要在15重新核为相同，透明复用既有阅读；本项补读初始化/自测及适用工具，重查命令、输出、来源和协议。两bank各自加权仿射可在固定任务下折叠，中间GELU及SW-SA/Haar/LayerNorm仍执行；同一已转换对象不能只改prompt更换任务，首次后bank和TSM已删。forward只接当前图和任务，无历史/未来/时间state。配置时和执行时的代价、任务切换与缓存多个配置的内存必须分开，不可把固定先验的执行期开销省去解释为整个pipeline免费。

公开工程层面的主要边界如下：

- README的inference.py不存在，真实入口为inference_3tasks.py/inference_5tasks.py；inference.sh所有例都是注释。实际5任务去噪sigma25、3任务50，CLI无sigma。若以后修命令/协议，应保存修正版，不能把改过的入口当作者原始复现。
- 训练set→list的one-hot顺序与推理固定顺序需历史映射确认；普通state_dict里未发现显式任务语义字符串。作者可能有外部环境或历史配置，当前静态证据不能证明其结果用了错映射。公开入口也未找到论文独立generic blind配置。
- train固定5维，而默认de_type只有3base；README150epoch/b8/4GPU与只暴露2张GPU、options默认1200/b2和scheduler15/100不同。数据重复倍数和外部ID名单影响实际预算，未由发布文件充分复原；没有执行定位默认首错误。
- 推理需RGB/255、像素域clip/uint8噪声、去噪先裁32及镜像pad下一个64，再unpad；metric是clamp的RGB float/skimage逐图平均。非去噪4任务实际都查input/target配对，并不提供无GT RTTS指标路径。
- 默认save_img=False且日志tofile=False，CLI未提供保存图像开关；仅设output_path不会自动留原图或逐图统计。完整实验还需保存命令/配置、stdout、输出与评价清单，不能从空目录/控制台均值推完整复现。
- 自测main只对随机输入运行未融合模型并计参数，未给部署后完整时延/FLOPs/内存脚本。固定包无FADE/NIMA/时间指标/顺序专家多pass、手机/TFLite/ONNX视频部署代码。清单层面缺少不等于作者历史环境没有它们。
- README标MIT但无LICENSE文本；requirements未显式列源码导入的scikit-image/termcolor。未安装或运行，不能据此宣称所有环境必失败或确定依赖解析首错。数据名字与公共权重存在也不能代替许可、训练split和重叠核查。

两份公开作者权重已完整归档，分别66,084,673/66,198,640B，长度及全ZIP成员CRC通过，metadata仅opcode扫描。可见未融合TSM/专家bank键名；它们是可得原件，不能再说权重完全不可得，也不能说张量语义、映射、精度或论文表格已验证。具体见[14恢复获取审查](../../../MIE-0014/derived/docs/resumed-acquisition-review.json)，旧错误回执与截断文件均保留。论文正式版本缺口和generic盲配置仍继承14的显式限制，本项不将其转成“正式版已复现”。

二期可参考的重点是条件配置的生命周期与可复核成本边界。优先核语义映射/固定任务等价/实际输入和metric，再把静态会话、逐帧条件与受限历史条件作同任务信息/同质量的控制；切换、错误条件、运动曝光和编码/I/O内存不能省去。当前代码没有实证支持稳定视频，也不能将同组参数融合组件本身当独立时间创新。后续仅在新实验分支Gate3开展数值/动态GT/设备验证，修正与原始路径分别登记。

本项适用公开材料的静态审查结论与缺口已列明；未来新增版本/完整授权/盲配置/正式版本或实验结果另追加复审，不改原件和历史结项。不提前综合125项。
