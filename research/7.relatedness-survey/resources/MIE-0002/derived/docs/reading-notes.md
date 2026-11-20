# MIE-0002 官方项目静态阅读记录

分析者：Codex  (sqs_A100)；日期2026-11-20；资源快照截止2026-11-17。先前MIE-0001结项后，顺序日志记录本项开始。本项为独立项目核查，承接前项的论文相关文件阅读并再次检查模型、主入口、验证与部署，不把文件抽取自动当阅读。

## 身份、版本与材料

官方论文链接到AVC2-UESTC/MobileIE，项目README标题、作者、论文链接与之吻合。固定SHA7030d125e28a0b9c30075f84f79f165acd13bdc5，tree1eac02be2a0155bc6e6b087c5d92fb11feb739bf；GitHub归档head记录提交作者账户名lukas，提交日期2025-08-05T08:44:00Z。不能把账户名推断为全部代码的唯一作者，也不把提交日当项目初次公开日。

原ZIP完整77文件，全部路径/大小/SHA见[archive-inventory.json](archive-inventory.json)，完整原件见[来源清单](../../sources/manifest.json)。56个可读文本文件抽取到自有缓存并由[源码清单](upstream-code-manifest.json)链接；**不是声称56文件全部逐行阅读**。第三方转换器的所有算子、历史build副本、通用测试和egg二进制不属于本次MobileIE核心机制全文审计范围，保留且明确不验证其完整性／正确性。

## 已实际阅读路径与发现

| 模块与文件 | 实际核查与边界 |
|---|---|
| README、LICENSE | 说明训练original、测试slim、UIE复用LLE配置、TFLite及AI Benchmark，含论文引用与联系信息。Apache-2.0是根LICENSE标识；数据和第三方代码许可不能自动等同。README的need_slims拼写与实际配置need_slim有差异；命令中的lle/isp为占位写法，argparse接受的是单个lle或isp。没有执行原命令。 |
| model/lle.py、isp.py、utils.py、utils_IWO.py | 全部核心结构已按原文件阅读，IWO用普通utils全文及完整差异核查。单帧RGB/4通道RAW、通道max、FST、HDPA；MBRConv运行统计融合；slim丢弃warm-up辅助尾，不丢FST。FST两个标量乘积与12维bias。普通模型未引入utils_IWO；IWO原语冻结投影weight而非bias，新增Xavier增量。没有运行融合等价或加载checkpoint。 |
| model/__init__.py | 由model_task拼类名、type选择Net/NetS；checkpoint采用strict=False，可能接受缺失／多余key，未来复现需显式检查返回缺失项。只在original且need_slim true时调用slim，README所述slim测试的need_slim并非都会触发转换。 |
| option.py、config/lle.yaml、config/isp.yaml | task及model_task选项；输出按当前工作目录创建experiments下子目录，没有递归确保父目录存在。默认model_task=isp、device=cuda；ISP1000轮/no-warmup、LLE2000轮/warmup10，与论文统述有别。配置路径默认并不直接指向ZIP中datasets示例树。 |
| data/__init__.py、lledata.py、ispdata.py | 读目录中文件并同名配对；RGB/255；RAW先2×2 RGGB打包再/4095。无排序保证、无序列／场景划分协议；训练shuffle，验证batch1。Dataset内部to(device)，搭配spawn和多worker会产生设备/进程问题的可能性，未通过执行量化。 |
| main.py | 直接设置CUDA_VISIBLE_DEVICES=5；train先warmup再Adam/余弦重启，validation eval及no_grad，按验证PSNR保存best并slim。forward_warm使用DropBlock与两个尾分支，并非仅低学习率原网络。测试和demo独立文件，不是视频循环。没有种子/实验清单/完整IWO二阶段切换。 |
| loss.py | OutlierAwareLoss有符号误差std/√2和detach；复合cosine+2×PSNRLoss。warm-up比较两个尾输出，RGB与RAW域按不同输出处理。训练目标与论文LVW公式非逐字相同。 |
| logger.py | 日志时间、等级、文件及stdout；不自动保存软件版本、GPU精度、数据hash或随机种子。 |
| complexity.py | 类名lle.MobileIES在固定lle.py中不存在；thop的macs变量叫FLOPs；合成随机张量；计时只有forward及末端同步，未包含完整数据/视频链。不修复或运行，不能报告复现时延。 |
| torch_to_onnx.py | 重新声明RGB slim结构、load_state_dict、opset12、固定[1,3,400,600]，无dynamic_axes。没有对所有输入域/分辨率公开导出验证。 |
| onnx_to_tf.py、tf_to_TFLite.py | 主路线用onnx_tf.prepare，GPU可见号硬编码0；TFLite允许BUILTINS和SELECT_TF_OPS。该文件未显式配置FP16，不能等同论文截图的GPU/FP16执行。 |
| test_TFLite_RGB.py、ISP.py | 默认Interpreter，无显式delegate/线程/精度；按文件预处理/调用/clip/保存。ISP只接受Bayer二维输入并/4095，不读取RAW黑电平、白平衡或相机元数据；没有首帧/切镜头/视频状态。 |
| onnx2tflite/setup.py、onnx2tflite.py | 可选打包路线；setup引用的readme.md和requirements.txt不在此ZIP相应位置，故不能假设原样可安装。wrapper用作者绝对路径及GPU2。现有egg不证明从源码重建成功。 |
| vendored converter.py、components/{__init__,builder,onnx_loader,output_check}.py、utils/op_registry.py | 静态读关键入口：ONNX清理/可选简化→Keras注册算子→TFLite，允许SELECT_TF_OPS；有可选fp16/int8，默认否。校验用全1输入，仅比较首输出/匹配形状，失败被catch为warning；转换返回不等于数值等价或所有设备原生算子支持。未读所有算子映射和generic tests，不称转换器完整审核。 |
| egg-info/PKG-INFO、requires.txt | 只核元数据／依赖和来源段落：MPolaris通用onnx2tflite，numpy≤1.24、TF<2.13/≥2.5等。不是针对MobileIE视频提出的新机制。 generic dependency不扩成独立MobileIE派生资源，代码作为本项嵌入材料留存。未安装或修改共享环境。 |

## 默认评价的精确口径

main.py训练验证和test都先对H,W做MSE，得到每通道误差，再对10log10(1/MSE)取均值；输出未clip再算PSNR，保存图像才clip并量化。其语义区别于先对RGB全部像素求MSE再计算PSNR，且不能从此推出原论文所有表格采用哪一套外部评价脚本。假设通道MSE为m1,m2,m3，则代码指标是−10/3∑log10(mi)，聚合指标为−10log10((m1+m2+m3)/3)，一般不同。这是源码与数学定义对照，没有模型实验。项目主测试未计算SSIM或LPIPS，故无法只用它重建论文全部质量列。

验证目录名eval15及按该目录选择best是一个需记录的协议，不能在未核实数据来源/划分的情况下直接定性作者发生测试泄漏。未来必须独立验证集及固定测试集，保留与原公开做法的差异说明。

## 权重及样例的静态文件核查

五个.pkl以zip目录和标准库pickletools的opcode方式读取元信息；**没有pickle.load/torch.load、import torch或任何网络forward**。FloatStorage总字节RGB各16300、ISP各16528，对应4075/4132个32位存储元素；metadata包含两处weight1/weight2/bias以及卷积/PReLU/attention键，和源码构造对账一致。见[checkpoint-static-inspection.json](checkpoint-static-inspection.json)。存储元素总数不替代逐张量加载校验、训练来源、精度图结构或质量证据，亦不解释作者4047/4104统计的原因。

pretrain还包括lolv1.onnx和lolv1.tflite，已归档并hash；未执行或断言与.pkl完全同权重/等价。没有为核查额外加载运行时。

12个附带样例是LLE、UIE、ISP各两对独立图片，头信息见[sample-headers.json](sample-headers.json)。实际查看[框架图](../figures/framework.png)和[样例接触表](../figures/sample-contact.png)：LLE两对600×400 RGB，UIE两对640×480 RGB，ISP两对256×256 I;16输入/RGB参考。RAW展示仅为可视化缩略图，不是训练预处理；源像素保持不变。没有序列标注、时间戳、完整划分或逐资产许可，不能当视频数据集。其视觉差异证明示例存在，不证明模型输出质量。

## 对后续工作的边界

核心单帧实现适合作为可审查的低参数基线；权重、图形和示例提高可获得性。要用于视频，仍需时序数据/协议、状态重置和完整设备指标。IWO接线、默认评价、strict=False、数据根目录/可见GPU/依赖环境都需在后续实验分支显式收敛，不能边调研边改上游或干扰服务器任务。

0模型执行、0训练、0GPU实验；没有证明默认命令能原样成功或报告论文数值复现。未知项不妨碍静态资源结论，但必须传给总体报告和下一阶段。
