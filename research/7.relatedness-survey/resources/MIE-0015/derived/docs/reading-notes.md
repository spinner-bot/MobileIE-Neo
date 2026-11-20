# MIE-0015｜作者项目逐路径阅读与复用记录

2026-11-21；Codex  (sqs_A100) <langxibielangle@qq.com>。14实际closed_limited后才begin15。固定f1d25d2、tree d914357c（2026-06-07提交，非首公开日）；21文件152472B与已结项14作者包逐SHA256完全相同，且新取确切tree对象的全部21个Git blob逐件一致，tree未截断。详见source-correspondence-audit.json与fixed-tree-receipt.json。

主张身份由论文source URL与README题名/8作者交叉确认，不能只凭owner。论文与代码同家族，不计独立数值确认。未安装requirements、未导入作者模块或执行任何入口，没有Torch/pickle模型加载、GPU、训练、Gate3。

适用核心阅读透明复用[14阅读记录](../../../MIE-0014/derived/docs/reading-notes.md)，按本项21件可读原件重新核对入口与关键行。复用范围包括DRMLP/TSM/SW-SA/编码解码、DWT/IDWT、训练/数据/配置/推理/metric；不是声称又做一次独立复现。新增实际阅读/复核包括DRNet_arch原1–76截断正态工具和585–597自测段（现在该模型文件597行范围完整）、inference.sh全文、README命令、requirements全部、transfer全文、data_util裁块35–46、metric_util全部、tensor_op保存路径36–80和util日志/种子路径1–100。未用的通用dtype/matlab/data/tensor工具不称全部逐行审计，custom-autograd数值正确性保持未知。

|路径与实际用途|项目检查结果|
|---|---|
|README 1–107|明确DRNet/TMM和作者；权重链接可用。训练暴露2GPU却要求4；所有推理例指向本包不存在的inference.py，sigma也非实际CLI参数。MIT badge未配LICENSE。承认基于PromptIR。|
|inference.sh 1–8|全部是注释例，不会直接启动评价；其中文件名inference_3tasks/5tasks才与发布文件相符，路径权重名DRNet.pth与下载3tasks/5tasks命名需要显式对齐。|
|inference3/5|显式CPU/设备初始化后加载未融合权重、逐DRMLP按原子任务折叠、cuda/eval，再逐图；无generic blind、视频流、顺序专家多pass或独立成本计时入口。参数cuda被用于CUDA设备，并非自动CPU后端。|
|net/DRNet_arch 1–597|截断正态为初始化工具且被相对位置表使用，不能称完全未用；末尾自测固定CUDA_VISIBLE=5，只随机图/提示forward并计未融合参数，没有switch_to_deploy、延时/FLOPs/内存完整基准。该段未运行。|
|net/net_utils|Haar DWT/IDWT的自定义autograd空间分解；其数值梯度/非偶数输入/后端精度未实测。原推理pad/裁块协议须保持。|
|train/options/dataset_utils|原子one-hot set→list与固定推理次序风险；默认3base对train固定5维；README150而默认1200、warmup15/max100；任务3/120/5/20重复与缺外部ID列表使预算需另核。源码他人绝对路径仅当字符串，不访问其目录。|
|inference5 73–142|非去噪任务统一需要input/和target/，dehaze按下划线前缀匹配PNG目标。正常4任务_get_gt_path均给路径，因此代码中的clean=None分支不提供真正无GT评价。不能直接冒充RTTS无参考FADE/NIMA结果脚本。|
|inference5 145–279、metric_util|逐图指标做样本数加权平均。默认save_img=False，CLI无保存开关，调用也没改；日志tofile=False，仅控制台汇总，没有自动导出逐图结果表/图像。拥有output_path不等于已保存输出或完整实验日志。|
|metrics/psnr_ssim 10–29|实际clamp RGB float/skimage/data_range1，不使用下方另一个Y/crop metric；没有FADE/NIMA/时间指标。|
|requirements/util/metric|README环境名及包列表是作者声明，未建环境验证。显式torch1.13/torchvision0.14、lightning2.2.5；scikit-image、termcolor被源码导入但未在requirements显式列出，可能有外部/传递依赖，不能据此声称安装必失败。util的torch._six是依赖具体版本的导入，未实际测试或泛指所有新版本。|
|transfer 1–21|从作者本地Lightning checkpoint取state_dict并去net.前缀，构造expert_dim3后strict=True加载再存普通state_dict；没有公开原训练checkpoint/任务映射，也没有转换盲模式。路径未访问，脚本未执行。|

两份公开权重在14阶段恢复完整并归档，逐SHA复用[获取审查](../../../MIE-0014/derived/docs/resumed-acquisition-review.json)及各自ZIP/CRC/opcode检查。两者各721 ZIP成员，meta中有TSM/专家bank键名，未见显式任务字符串；没有按张量反序列化确认所有形状/语义，也不能证明历史训练映射。没有把之前错误success/截断原件当完整权重。

项目没有新增图像/视频示例素材；README无科学定性媒体，已有14论文图表阅读属于同一研究载体的既有证据。21文件清单无dataset原始图/外部ID文件、原训练日志、TFLite/ONNX/手机应用、独立计时/FADE/NIMA/时间指标脚本；这个结论仅限固定发布包，不断言作者所有环境从未拥有它们。

适用结果：当前公开核心足以审查会话配置/折叠、输入尺度、任务语义和协议接口，不能直接拿命令成功或checkpoint完整作论文全部表列/视频/手机复现的证明。后续实验首先保存版本/依赖/任务映射/修正命令与metric协议，修正版独立标记；再按原图尺度、同质量/同信息和完整成本做控制。原刊全文与generic blind对账仍由14证据受限记录继续保留，不因代码条目审查而消失。
