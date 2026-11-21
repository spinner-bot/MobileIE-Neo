# MIE-0114｜ULightIF作者项目：原发布与重构入口

2026-11-21；Codex  (sqs_A100) <langxibielangle@qq.com>。有效阅读序22，59关闭后才begin。0外部源/模型/反序列化/训练/GPU/Gate3。

固定bc952b6f1f18a6480628fc91f6ce96a93484817c/tree14355b201e2309299f61a175211b9b0c8f4d2648、2026-09-02提交。18件1025443B与59原件/本项独立ZIP清单/官方Gitblob逐一一致；3PTH CRC再次核，13适用文本的实际全文阅读和框图/操作码核查透明复用同SHA59记录，不计独立实验。14文本中unused430行scheduler仅查清单/全包无导入，实际调用torch.StepLR，未声称逐行读通用库。全部原件在本项artifacts，固定读取副本在upstream-code，5份先前阅读/视觉/结构证据摘要保留；59不回写。

README原文明确这是ULightIF官方实现、主要基于MobileIE，标题/三作者/DOI与59核。readmeack的MobileIE链接是相对路径，包内无该文件夹，但显式名称及MBR源关系仍有证据；不能把相对链接当已取其它官方仓库。59正式全文、文内语境及历史数字缺口保持，本项不解除。

## 完整当前源的行为与边界

build.py用MBR3/PReLU构建IE24、IF16、ID32，net.py包含reflect边界、多卷积/BN拼接和running统计折叠；build_slim.py对应当前标准Conv/PReLU版本。当前网络未保留原作FST/双注意力，也没有时间记忆。IE输出illum用于VIS除法；IF共用encoder处理VIS/增强VIS/IR三个亮度，双decoder得到fu/fuEn；ID(fu)输出R/L。两模态当前图像，没有历史/未来/时间对齐或状态。6145/14305/39204与三权重FloatStorage对应，合59654；形状算术/操作码不当运行计数。

test_p.py定义原论文slim类、加载weights内3个原pth，10次warmup后执行三网并存五输出。末注常规fu只统计IF、低光fuR统计IF+ID；实际默认计数/计时仍三网+辅助增强分支。固定无状态网络下fuR所需依赖为原VIS/IR→fu→ID，teacher IE/fuEn路径可单独研究裁剪，但未改/跑或测等价。参数口径、实际执行次数与完整预处理/保存成本必须分开。文件数不等于序列FPS。

test.py与原pth入口不同：加载原包未包含的三stage Lightning last.ckpt，然后调用slim。它提供check_fold合成输入差异打印，但没有容差assert；IF比较是原网Y输出，对照slim RGB输出再转Y。若RGB重建经过clamp，roundtrip并非纯Y恒等，即使卷积折叠精确也可能打印差异；这里只是函数组成的条件推断，没测差值或断定折叠失败。options的check_fold/calc_flops/enable_timing没有bool转换；字符串False不等于关闭，main还固定enable_timing=True。未使用PLIF.forward若给RGB落入1通道原IF的条件问题不倒说实际训练必失败：training_step转Y，ID阶段使用内部转Y的slim。

两入口timer在读取/拷贝调用后、保存前，同步网络计时不含逐图归一化/量化/写图；只有test_p显式10warmup。count_flops将thop的MAC/G标GFLOPs，非卷积/色彩转换/内存/完整端侧预算未认证。CPU/CUDA回退不证明移动NPU或手机时延，没提供导出/部署/功耗/热稳态材料。

ImgRGB_save/ImgGray_save按**每张输出的全局min/max**映射到uint8，非简单clamp，无epsilon。batch1时极值改变映射可影响全图，恒值有零分母的条件算术；不能据此称已观察视频闪烁/NaN。未来要分别保留原tensor、色彩转换后、保存后序列，等质量比较网络与后处理贡献。本阶段没生成任何新输出。

## 当前重构训练和数据

train.py三stage分别fit同一paired trainloader，save_last而无验证loader/best质量选模；默认各10epoch/batch8/crop96/Adam1e-4/StepLR50/.5/seed3407，仅是当前默认，不是历史论文重建。PairDataset同名IR/VIS同步crop/翻转/旋转，目录没有扩展过滤/显式配准核验，原图需能容纳crop；没有序列划分/时间GT。默认/root/gpufree-data和旧timestamp ckpt路径是作者环境，未访问这些目录，也未改本项目全局配置。

IE输入统计曝光目标/空间/平滑，IF强度/梯度/SSIM，ID重建+fuEn亮度/冻结ImageNetVGG19感知+棋盘下采样一致性，ID还训练加sigma15/255噪声。它没有正常照明GT，却有成对IR、合成噪声及外部预训练先验；不称无数据依赖或时间一致性训练。VGG只在训练/重构加载路径，原test_p不计它为部署神经网，未下载/执行。

README给MSRS/LLVIP/M3FD官方数据链接，包不含数据/原划分、历史训练log、三last.ckpt、requirements锁或LICENSE正文。记录实际可得性，不推法律授权/不授权或安装必败；也不把当前源码完整等同所有数值可复现。项目的“reproduce paper”是作者对test_p的用途描述，本次没有运行来验证。

## 用途与未知

当前全部公开载体的适用静态分析足以闭环：可定位单图三网、辅助teacher/推理依赖、原/重构入口与归一化行为。可交接的研究问题为等信息/等质/同预算下的辅助训练和因果部署、网络/色彩/保存三层扰动。额外IR不能混作单RGB视频改善；空间下采样不当时间邻帧监督。

59的正式全文/完整图表/引用语境、历史训练与三权重/表列对应、数据许可/逐名划分/重叠、独立指标/时序/全手机成本仍未知。以后合法新证据追加复审；实验仅待另行Gate3任务，本项不预断假说有效/新颖，不把59/114算两次独立实证。
