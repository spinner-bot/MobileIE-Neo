# T1完整已知时序蒸馏对照：同F0起点、推理不加算子

2026-11-20T17:25:32.906350+08:00；Codex  (sqs_A100)。依据原训练目标预检四组通过后的独立登记；本文件先于新学生训练。它是必要已知KD/训练期时序参考，不是创新方法；V1及全部旧STOP不解封。代码与本协议须在CPU/CUDA资格或正式更新前另冻，当前没有已完成训练或DEV效果。

## 单一问题与三个独立控制

能否在保留原增强模型/参数化/初始化的条件下，仅改变训练监督的信息来源，将固定强post的一部分收益移入单帧映射。每seed0/1使用同seed R44 square_mse原4075参数F0；它已有官方LoL及项目8K历史，三个学生直接deepcopy同一完整权重，不建随机新骨干、不追加随机门或改变推理。教师冻结eval，三学生相互独立、完整参数均可更新。

- gt：MSE(S(x), GT)。
- spatial：.5 MSE(S(x), GT)＋.5 MSE(S(x), F0(x).clip[0,1])。
- temporal：.5 MSE(S(x), GT)＋.5 MSE(S(x), T4_DIS(F0(x).clip[0,1]))。

T4_DIS严格复用预检原FixedDIS/RecursivePipeline，原生OpenCV warp/原FB接受/alpha.5，窗口首帧复用raw，其后最多三历史；教师运动只读F0当前/前raw输出，GT不生成teacher。四帧窗之间重置，完整视频的原强post参考保持连续历史，这一差异明确披露。等权MSE是已有监督/教师混合，若残差两边减同F0又等于输出KD，不另取新名字。这里不声称复现OFD风格网络或PocketDVDNet架构，只适配已知输出KD思想。

学生loss使用未clip的raw输出、teacher clip后为浮点[0,1]；评价clip[0,1]。GT-only不额外隐含teacher项或temporal loss。没有蒸馏feature/flow、GT flow、variance、Jacobian或rank附加项，也没有alpha/lambda/噪声/结构网格。

## 固定数据、优化与完整预算

只用原MixedPool B4T4/128、2 DAVIS合成＋2 OLiVES真实，沿用原排除、翻转与噪声；每seed数据步8000..11999，三个学生共享完全相同窗口。每学生每seed4000更新，三组×两seed共24000正式更新。三个学生各fresh Adam1e-4，(.9,.999)/eps1e-8/wd0/globalclip1、FP32/TF32关、无调度/AMP/早期选模/追加预算，只有4000唯一最终参与DEV，1000备份不排名。与原MBR训练的优化参数化不同，不能写成复现原R44训练。

每个共用数据步teacher只算16图一次，得到所有三组目标后各更新一次；教师/学生/Adam状态不跨组共享。两seed正式合计teacher128000图、student384000图、teacher DIS96000双向对/192000calc，GT-derived参考flow正式训练0。共享准备不等于三组相同FLOPs，GT-only单独运行本来不需teacher，要并列披露训练计算。

## 实现资格与共享GPU

先CPU仅seed0、step8000/8001，三学生各2次丢弃更新，共6；相同数据在CUDA再6，全部丢弃，正式重新从原F0初始化。每次teacher32图、student96图、24双向DIS对，单列不混正式计数。只核finite loss/梯度/参数、三组相同原权重起点、teacher无梯度、首帧teacher恒等、4075参数/输出形状、最终保存重载有限；不新增bit级Adam回放/无边界人工网格，不把资格当效果。

GPU使用现有进程前UUID绑定入口，真实CPU准备后内层新11×2秒采样/项目锁/实际PID UUID核验；统一选择同组4–7、最多一个正式GPU worker。预估进程4GiB/allocator2GiB、既有任务8GiB增长＋1GiB余量、mean util≤60/忙样本≤25%和原25%合作占空/逐共同batch监测不变，不操作他人任务。资源暂停只在三组该共同step均完成边界保存所有模型/Adam/RNG/步号后75接续；不丢更新或科学失败自动重试。每1000共用step保存三组完整状态。没有墙钟截止；长等待按实际速度估算sleep减少激活，醒后核原句柄。

## 唯一最终全量评价

原10条件/seed（六SYN64帧＋四ND16帧）、同GT运动接受域/CPU2/OpenCV1，各学生raw＋自身half-FB.5＋自身DIS.5，三组×3输出×10×2=180新质量/8064逐帧；新学生骨干2688图。强post各自只用自身预测/自身raw对应与递归历史，不借GT部署。单帧学生推理不读teacher、flow或state，4075参数/3828HW+144卷积MAC与F0相同；这说明算子预算不增加，不是实测手机速度。

全部180预测以相同官方ColorVideoVDP0.5.7/standard_fhd/名义30fps评分，8064比较帧；复用原R86 F0 raw/half-FB/DIS60条件，合240共同条件/48域seed均值/12输出。公开全部PSNR/SSIM/Etemp/JOD，不只挑SYN、均值或某seed。四预定联系图为seed0/1×bear_photons20/video47首/中/末，列Input/GT/F0/F0-FB/F0-DIS/gt-raw/spatial-raw/temporal-raw/temporal-DIS；必须实际阅图，静态图不作MOS。

## 原则性资格、止损与解释

时序KD学生先保护相对原F0 raw的恢复：每域每seed PSNR≥−.10dB/JOD≥−.05、逐条件最差PSNR≥−.5。主要强参照八个：原F0、gt、spatial、temporal四套权重各自half-FB和DIS输出；另列gt raw与spatial raw以区分额外GT适配和普通锚定。十参考每域每seed保护PSNR≥−.05/JOD≥−.03，逐条件最差PSNR≥−.5/JOD≥−.15。

A需要相同质量保护且十参考Etemp均≤.90。B需要相同保护、对八post Etemp≤1.02，并对两个raw控制Etemp≤.95；保留推理成本取舍，不冒充时间10%优势。只有A/B任一质量成立，才另登记原CPU2全部条件/seed16帧prefix、warm2/正式3次、顺序交替的完整pipeline成本；A需成本≤最强对应经典1.05，B≤.80，手机/外部另需实际证据。普通KD即使通过也只是可用基线，不自动形成独立创新。

若保护或A/B失败，停止这套固定F0/T4-DIS/等权KD的teacher、alpha、T、混合比例、loss、LR、长度和seed局部网格，不扩失败候选成本/外部/手机；保留所有条件正例，不否定一般时序信息或所有训练-only方案。训练目标预检不等于此门槛，也不能绕开它。始终保留原84表值/PDF020和独立最终封存，整篇论文goal active；main/自有data/指定作者/no-ff/逐文件提交、阶段推送、7号与其他worktree边界继续。
