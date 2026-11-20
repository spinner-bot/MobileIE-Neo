# MIE-0005｜Ascend NPU 项目阅读过程

作者：Codex  (sqs_A100) <langxibielangle@qq.com>；实际阅读日期：2026-11-20。静态资料分析，0模型执行、训练、NPU/GPU或设备实验。

## 身份、版本与范围

原件为[归档回执](../../../../collection/receipts/sourcezip-lhfgghc--LLVE-AMNet-NPU.json)对应固定提交 `4c535f6462f5be9669268e35bb7fd2d6010be3ee`，tree `a6c9af8b21a525e8236601441af31f4693b01027`。提交元数据日期为2026-06-14T13:24:01Z，消息update details；这不等于初次公开日期。MIE-0004固定README指向本仓库，本README反链原项目并称Ascend NPU implementation；不把账号当所有论文作者。

[完整清单](archive-inventory.json)69文件，[文本清单](upstream-code-manifest.json)40文件。与MIE-0004的固定快照逐路径/摘要比较为33相同、6改动、30新增、8未出现在本包；这是两个快照的内容差异，不证明Git祖先关系或移植先后方向。参见[逐项比较](upstream-comparison.json)和[完整文本差异](source-diff.patch)。模型、五配置、五划分、数据类、scheduler、checkpoint、logger、五launcher等相同内容复用MIE-0004适用核心阅读；新推理入口全文和全部六处文本改动独立读过，并重读关键上下文。不是全仓逐行重读声明。

## 改动逐项

- README只有Ascend身份和一个图片目录推理命令，没有安装/设备/CANN/torch_npu/PyTorch版本、时延、内存、精度或功耗表，也不含本地权重。
- train.py:137–145优先导入torch_npu并检查可用，再退CUDA/CPU；仅捕获ImportError。原数据构造、freeze/AdamW/scheduler与训练调用保留。
- trainer.py:127–133允许cuda/npu启用GradScaler，319及452/466/479/492的autocast从硬编码cuda改用self.device.type。五份原样配置实际均use_amp:false；“增加AMP接线”不能写成“默认启用NPU FP16”。
- seed.py增加CUDA可用检查及torch_npu/NPU seed分支；不等于全部算子确定性保证或完整随机状态恢复。
- metrics.py仅DistillNLLLoss在参数/target缺失时的零张量默认设备cuda→cpu；没有改PSNR/SSIM或给主训练加入NLL。temporal_loss仍只是工具定义，默认主路径未接入。
- .gitignore取消outputs/忽略；不是算法或性能优化。

## 新的目录推理入口

逐行阅读[完整入口](upstream-code/scripts--inference_full.py.txt)：17–35固定latent256、hidden256、单层ConvLSTM、decoder块[2,2,1,2]。未显式传event_enable/ir_enable，沿用模型默认True；只提供RGB不代表跳过两个隐式模态生成分支。

42–53：自动选择先torch_npu、再CUDA、再CPU。显式--device直接返回torch.device(pref)，跳过该函数内torch_npu导入与可用检查；没有执行，兼容性与错误行为保留未知。

56–70：默认checkpoint路径是checkpoints/amnet_pretrained.pth；归档中没有checkpoint。torch.load(weights_only=False)后取model键，首键module.前缀触发去前缀，strict=False仅打印缺失/多余键数量；不核完整键覆盖、来源哈希、EMA选择和版本兼容。本调研没有反序列化或加载它。入口不含缺失文件时的友好回退；与训练器“预训练缺失后继续”的条件路径分别记录。

73–90：只glob小写*.jpg并按文件名字典序排序。全部RGB PIL→ToTensor保存在列表，再stack为[T,3,H,W]搬到设备；不resize、不pad、不检查帧率和时间戳。相同尺寸是stack必要条件；任意尺寸/长度的实际可运行范围未测试。调用inference_video后才逐帧转CPU、clamp、乘255并转uint8，以001.png等重编号保存，不保留原始文件名/元数据。目录中的PNG/JPEG/MP4不是此glob的输入。

93–129：先加载模型，再遍历所有子目录；每子目录一次infer_clip。没有摄像头、视频解码/编码、同步计时、热身、峰值内存、功耗、量化/导出或移动运行时入口。本目录范围内未见这些完整pipeline，不能外推作者所有私有工具均不存在。

## 因果、时序与成本的区分

模型与MIE-0004字节相同，amnet.py:149–283每次离线调用清零H/C，按t向前递推；383–393设置eval并返回整个序列。RGB模型计算没有显式未来帧索引，但示例读完全部帧才调用，返回前也缓存全部输出；这两个事实可以同时成立，不能把此入口称常数内存在线流。预取/读取未来文件不等于单帧预测利用未来视觉信息。

395–479另有start_stream/stream_step/end_stream，H/C可跨调用保留；本示例没有调用。stream接口要求调用者处理eval、状态清零、场景切换、分辨率改变及时间戳。原始事件体素的真实时间截止仍不由此RGB演示证明。

静态算术：输入7张1920×1080 RGB FP32连续张量为174182400B=166.11328125MiB，仅x；同样大小的输出、CPU帧列表、临时激活/aux_record不包含。默认H/8×W/8、256通道的一层FP32 H/C为66355200B=63.28125MiB；单ConvLSTM cell构造参数4719616。这些不是全模型参数、设备峰值或测量时延。AMP配置False，推理入口也无autocast/half/量化，不能声称默认FP16成本。

S2DG仍有FFT/complex/iFFT（adapter.py:133–139）、attention、ConvLSTM与动态状态；未看到针对这些的后端替换或分区说明。设备字符串改变不足以验证目标NPU支持、CPU回退占比或精度。没有断言必不支持，也没有执行兼容性试验。

## 配置、评价和论文关系

五配置按摘要完全复用原项核查：DID batch8/lr1e-4/warmup5，SDSD indoor 4/1e-4/5，SDE in/out与SDSD out 4/1e-6/10；100epoch、clip8/stride2、base256/crop128²。原训练和RGB/真实事件条件不是推理入口实际读到的权重信息。

test loader仍batch4，trainer.py:448–456仅y[0]/x[0]，默认按crop/重叠clip且每clip重置状态。PSNR仍全RGB MSE后log，与MobileIE默认逐通道PSNR口径不同。缺预训练无resume时仍222–227警告返回而外层打印loaded；冻结范围仍需按配置区分。指标JSONL写出仍在，不能因Logger空实现宣称无日志。详细继承限制见[MIE-0004结论](../../../MIE-0004/derived/report/analysis.md)。静态风险没有证明作者发表实验走这些默认路径，更不证明数值造假。

本项没有新论文实验表。不能把MIE-0003/4作者报告的质量数字重新归为NPU实测，也不能将本项与原项目当两个独立科学方法/验证。预训练多模态信息条件、数据内容/划分重叠、论文与公开loss/基线评估的差异沿用原项明确未知，不由移植名称解决。

## 附件与实际视觉审阅

[视觉记录](visual-review.json)：单页framework PDF文字读完、渲染实际查看，图含optional event/IR、S2DG/FFT、H/C递归与残差解码及L2监督标注，无硬件/时延证据。最初可选fitz导入失败、系统PDF工具不可用检查未处理资料；随后用既有项目pypdf/pypdfium2成功，无安装/环境修改。

28张归档图像分为video1/video2各7输入JPEG及7输出PNG，均1080p；[两张接触表](demo-contacts.json)已实际查看全部配对缩略图。室内墙/红色设备及展板走廊，作者输出明显更亮、相邻取景轻微移动。没有GT、生成设备日志、时间戳/帧率或匹配权重哈希；不能据静态表判断闪烁、纹理真值、手机速度或实验重现。未自行生成增强结果。

## 价值、反证与缺口

本项是同作者关联的异构后端接续证据，适合用作端到端部署证据要求和状态API的参考，不能当MobileIE被替换为轻量骨干的先例，也不能给新方法添加独立重复计数。对二期的可证伪问题是：相同权重/输入/精度/状态策略时，离线与显式流式能否保持输出一致并降低长序列缓存；固定目标SoC上的FFT/状态路径是否达到所需画质、延迟、峰值内存和能耗。所需控制为逐帧MobileIE、同权重离线/stream、清零/保留状态、完整warmup/同步/I/O拆分和切镜/运动/曝光场景；全部留给后续实验分支。

未解：NPU具体型号与软件组合、二进制权重/覆盖和实际生成设备、算子支持/回退、量化/导出、完整预训练配方、GT和帧时间、端侧连续质量及总成本。69文件无LICENSE/requirements/锁定环境，README也未给许可；原项目MIT声明不能替代本包授权核验。可分析源码材料完整不等于部署或许可已验证。
