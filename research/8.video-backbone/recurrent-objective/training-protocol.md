# T2完整已知控制：clip-GT与部署DIS递归最终GT监督

2026-11-20；Codex  (sqs_A100)。先于任何新优化更新登记，依据零更新实现资格通过。它是已知递归训练的适配/较强基线，不是新方法；不修改T1/V1及全部旧STOP，不把4075或推理零状态误当用户硬约束。本协议、全链代码须在优化资格/正式更新前另冻。

## 两独立模型、固定推理函数

seed0/1各从同seed R44 square_mse完整4075参数F0 deepcopy两个学生，完整参数独立可训练、无新核/描述/门。clip_gt：MSE(clip(F(x)),GT)。rec_gt：MSE(R4_DIS(clip(F(x))),GT)。R4每个B4T4窗重置，首帧raw，后续原native DIS ULTRAFAST/原FB接纳/原OpenCV1/32插值/.5递归上一fused RGB。flow只读本模型前/当前raw，field与mask stop-gradient；RGB历史窗内不detach。GT不用来造field，没有KD教师/辅助loss/flow监督，也不称完整DIS可微。

原生前向/历史梯度已用固定两训练批核验。正式训练只rec_gt估field；clip_gt单独逐帧loss。两组只共享相同窗口，不共享权重/Adam/field或教师。clip-GT控制隔离与T1未clip loss的截断梯度差异；训练计算并不相同，须披露。部署仍逐帧F一次，完整主输出rec_gt+原DIS，包含状态/warp/转换，参数只计NN4075，卷积3828HW+144，不能称零额外开销、手机速度或相同完整FLOPs。

## 原预算与资格

沿用MixedPool原排除/翻转/噪声、B4T4/128、2SYN+2真实，seed0/1数据步8000..11999。两组各4000×两seed，共16000正式更新/256000学生NN图，rec_gt共96000双向DIS对/192000calc；clip_gt训练0field，无教师前向/GT-flow。各fresh Adam1e-4/(.9,.999)/eps1e-8/wd0/globalclip1，FP32/TF32关、CPU2/OpenCV1，无调度/AMP/墙钟截止/DEV选模/追加预算。只有4000唯一最终评价，1000整共同步备份不排名。

先CPU seed0数据8000/8001，各模型2共4丢弃更新；再CUDA相同4，全部丢弃，正式重新从原F0初始化。每backend64学生NN图/24DIS对、teacher0。只核起点相同/4075、finite loss/梯度/参数/梯度范数、rec首资格批原生前向≤2e-6/首帧恒等、最后保存重载finite，不追加bit级Adam或探针网格。原零更新资格32NN/24DIS/4raw梯度/4参数梯度另列，不混优化资格或正式计数。

沿用进程前UUID绑定、真实CPU准备后新11×2秒采样/项目锁/实际PID UUID确认，统一组4–7、最多一个GPU worker。预估整进程4GiB/allocator2GiB、原任务增长8GiB+安全1GiB、mean util≤60/90%忙样本≤25%，逐共同batch监测/25%合作占空。只在两模型该共同step均完成的边界保存全部模型/Adam/RNG/计数后资源75接续，零新更新也须转交原checkpoint，不丢步或重复更新。科学失败不自动重试；正式CPU未登记。长等待按实际速度sleep并核原PID/start/runtime，不按估算截断预算。

## 唯一最终全量评价与比较

原每seed十条件（6SYN64帧+4ND16帧），各学生raw/自身half-FB.5/自身DIS.5，2×3×10×2=120新质量与官方ColorVideoVDP0.5.7/standard_fhd/名义30fps，均5376逐帧；新学生DEV1792 NN图，两个post各1752双向对。每seed60行/2688帧/896NN、各post876对。推理使用原RecursivePipeline，训练Torch固定场对应只用于反传，不替换原评价算子。所有指标PSNR/SSIM/Etemp/JOD完整公开。

复用T1全部240条件（含原F0的60）共360/72域seed均值/18输出，不再重新推理T1；全部来自重复DEV，不能称独立视频/最终。主候选为rec_gt_DIS05。11强post参照为F0、T1的gt/spatial/temporal、T2 clip_gt五套权重各FB和DIS，加同权重rec_gt_FB05；所有raw也列在全表，不只挑SYN、最好seed或有利post。

先保护主候选对原F0 raw每域seed PSNR≥−.10/JOD≥−.05、逐条件最差PSNR≥−.5。对11强参照每域seed保护PSNR≥−.05/JOD≥−.03、逐条件最差PSNR≥−.5/JOD≥−.15，44个主要比较全保留。

A：上述保护全部通过，11参考Etemp均≤.90。B：保护全部通过，11参考Etemp均≤1.02，另对原F0 DIS及独立clip_gt DIS四组均≤.95（8个额外学习增益比较）。只有A/B任一质量资格成立，才另登记完整CPU2十条件/两seed16帧prefix、warm2/测3/次序交替的流水线成本；候选相对F0 DIS和clip_gt DIS均≤1.05才有同流程预算的采用资格。B不是T1“单帧省20%成本”的旧判定：此问题部署保留DIS，仅检验同流程下的可学习改进，绝不回改T1失败。全部资格通过也只是更强已知优化参照，不自动形成创新或手机结论。

四固定联系图seed0/1×bear_photons20/video47首/中/末，九列Input/GT/F0raw/F0DIS/T1gtDIS/clip_gt raw/clip_gt DIS/rec_gt raw/rec_gt DIS，必须实际view_image，静态图不是播放/MOS。新方法创新、外部自然动态、手机和独立最终仍须分别建立。

## 固定止损

若恢复或A/B任一完整资格均未通过，停止固定原F0/native DIS/.5/四帧递归监督的alpha、mask、插值、loss、LR、训练长度/seed局部网格，不扩失败候选成本/外部/手机。保留所有条件正例与单帧/自身post/已知KD参照，不否定所有递归训练。旧T1/V1/R28等全部STOP/原84值/snapshot003/PDF020/独立最终封存保持。main、自有data、指定作者/no-ff/逐文件提交和阶段推送、7号与其他worktree边界继续，论文goal active。
