# MIE-0013｜FGDPA作者实现独立资源分析

2026-11-20，Codex  (sqs_A100) <langxibielangle@qq.com>。前12项close后才begin13；本项完整21件/13可读文件（含Apache许可证），不执行上游源码/模型/测试/训练/GPU/Gate3。

全部原件SHA与前项论文作者证据逐一匹配。前项已完整静态读README/config/model/uie/utils/model入口/data loader/uiedata/loss/main/CLI/logger/test名单；本项复用这些相同字节的阅读证据，另重核模型入口、默认配置、DCT初始化/实际调用和full/slim前向/转换。许可证只核Apache条款类型与模板署名尾，不声明逐行法律审核。日志1000epoch仅解析和首末/最佳/学习率核；checkpoint/5pyc仅头/ZIP/摘要，未经反序列化。作者定性PNG与此前实际完整观看相同，不制造第二独立视觉/科学实验。

## 身份/版本/材料

530c692feb…，2026-04-24T11:55:01Z，dbb0a…tree；README论文作者Ao Li拼写按原件，MobileIE官方链接和partially built upon明确，不只通过星标/搜索命中认定关联。原件有训练实现、slim checkpoint和90个test名，不能再称“只有摘要/没有任何代码或日志”；但缺requirements/完整运行环境、历史配置与权重哈希链、设备benchmark/导出脚本，公开实物存在不等于结果已证实。

## 实际前向/折叠/时间条件

uie.py17–46先保证可pool尺寸；不整除先bilinear到较小可整除，再avgpool，任一边<32返回原图。固定32²是典型尺寸协议不是任意尺寸绝对不变。FFT/logabs/GAP/均值归一与当前图空间GAP组合，另一分支当前图max/mean，最后alpha/beta/lambda混合与F相乘，无历史/未来/外部motion/场景切换处理。FST两标量与通道bias的平方形式保留，BN runningstats不称时间恢复记忆。

空间分支MBRConv1(2,C)→slim Conv1×1(2,C)，故As实际C×H×W；不是正文另处单通道图。Fig中3×3不从名称推实现。C12尺寸字面4234参数与论文对应，不是checkpoint实测；attention的全分辨率As、A和中间特征仍占内存，FFTguidance非零推理成本，静态可学门控也不是输入无关attention。

utils.py的MBR5原始/BN双路concat、MBR3八路加DCT第九路concat再PW，MBR1双路concat，slim折BN/核padding/偏置与融合PW；不是所有层只简单求和。DCT固定DW/PW/tanh门控线性可折，不把非线性/FST/FFT抹去。uie.slim复制FST/PReLU/gates并映射卷积到目标state，只有存在目标键才替换；未运行fold对账/设备精度验证。

活跃MBR3→DCTLinearBranch默认PW/gamma双零；此前[公式推导](prior-dct-initialization-review.md)完整保留。本项确认配置pretrainedFalse、fresh Adam且源码没有额外非零重置。有限梯度/未加载非零state下两梯度零，分支不贡献信号；这是条件静态事实。未调用DCTAddIntoConv类不当默认实际路径。历史训练用其它源/状态可能不适用，不能从当前固定实现断言论文消融虚假或整个网络训练失效。

## 默认入口/数据/损失/运行证据

model/__init__.py按type选original或NetS，pretrained仅truthy才torch.load，need_slim仅original+True转换。当前yaml默认original/pretrainedFalse/need_slimFalse，因此README test/demo命令不会自动加载所发best_slim；还需按真实格式配结构/权重，不以strictFalse当完整正确加载证明。未反序列化文件确认内部键，不能承诺任意设置都可运行。README有FGDRA仓名/need_slims与实际need_slim不一致、requirements文件缺失，按实际源码存限制，不替作者改文档或借用共享环境安装。

UIEDataTrain sorted目录、RGB/255、尺寸不足resize、256随机crop，无flip/rotate；valid/test原分辨率、不crop；DataLoader valid/test batch1。90唯一test名只说明可用名单，当前代码不读该文件/自动构建800/90。不同split用现成权重会混协议，README已提示重训。没有下载数据逐样本/序列/训练重叠/许可核查，图像切分不当视频独立泛化。

loss OA采用signed residual/std/sqrt2、detach权重，叠余弦和2PSNR；是同图GT非帧间监督。main每epoch整套valid挑best，同时可保存slim；训练1000/b4/seed3407，warmupFalse，lr硬编码.001而作者log写.0009。1000条log最佳epoch657约23.969，末22.9797，不能忽视存在日志、也不能将其当已证源/权重/表完全同版本。默认测试仅PSNR，无运行速度/时序/GT视频评测；demo Dataset返回三项而loop只解两项，且gt=None也进入默认collate路径；这两处都是静态接口兼容风险，未运行定位首个实际异常。

setup_repro设deterministic/TF32，不代表当前环境/FFT后端已确定可运行/可复现。源码无TensorRT/手机/Jetson benchmark链，628.25GPU与106.9AGX64GB均原论文条件声明，不能据包名或4K参数证明手机快、全pipeline低功耗。

## 可操作价值与后续控制

这是真实公开的MobileIE继承实现，普通FST/频谱引导/训练折叠均有先例；本项不另计为独立确认科学结果。优先后续补原始训练版本/初始化/state与split/检查结构加载，再核fold数值/临时内存/FFT后端和真实video的逐帧统计敏感性；若修初始化或demo接口，必须标独立修正版，不替代原作者证据。后续比较应同空间质量、历史信息/运动曝光/遮挡/编码和完整设备成本，当前候选未执行，不回滚主线已冻结探索或开始总体综合。
