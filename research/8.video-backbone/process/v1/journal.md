# 主题8 / V1过程

2026-11-20；Codex  (sqs_A100)。所有时间Asia/Shanghai。原R88全闭合且不重跑；7号留给其他分支，main工作，demo/org与survey/pre独立worktree不修改。

## 13:02前：从原理阅读转为完整实现

核实际main=26300536、干净且ahead1，/data已挂载，项目规范和SSH隔离有效；无根据旧live提示重启任何任务。此前只确认编号的目标轮为no progress，本轮采取实际源码/证据更新。

分别提交原模型/真实F0继承审计、直接先例/唯一完整系统、骨干、原生因果流水线、完整预算/七组强参考协议、运行辅助、八丢弃资格、两阶段三控制训练与完整质量评价。代码审查时在运行前修正整除尺寸免padding及resource续接不覆盖已训练控制；不作为科学结果。六模块py_compile正常。34个源码/权重条目和协议已冻结于source-freeze001，后续资格/训练/质量期间不改；JOD/render/summary之后另冻，不可将准备称完成。

原文阅读边界：MobileIE表1的1120FPS属于GPU列倒数，SoC6.72ms分列；F0是官方LoL＋R41 4K＋R44 4K，R40不是权重祖先。FFDNet/EMVD/LPS有直接先例，普通重排/递归/门/KD没有创新主张。两次作者/CVF PDF获取失败，改读原文HTML；一次TARONVD arXiv编号误指通信论文，识别标题不符后丢弃，没有采用该页结论。未下载/运行外部模型。

固定V1共28000正式更新，含两seed各2K教师暖启动＋3×4K GT适配；正式尚0。新空间16180/门后16365参数，MAC为解析而非速度，候选保留经典DIS、两遍CNN，不能称4K/零开销。四域-seed空间保护与完整强参考质量/成本止损、所有旧STOP和最终封存保持。

## 13:02:17：原唯一CPU资格实际启动

命令：`CUDA_VISIBLE_DEVICES='' ./scripts/project-python research/8.video-backbone/qualify_v1.py --device cpu --attempt 1`。exec session74545；实际子PID2980317（系统ps起时13:02:17），解释器位于zrh自有data。13:02:25核原PID仍在，尚无config/更新或runtime，仅准备阶段；stdout当时为空，不称通过。预定仅seed0 step0真实训练窗，8更新全部丢弃，不继承。之后核同一session/原终态；CUDA资格/正式启动尚未进行，不依观察超时重启。

## 13:02:46：CPU原唯一资格正常结束

原session返回0，runtime正常27.587185秒、CUDA_initialized=False；8丢弃更新、377骨干图、32教师图、120双向DIS。奇数127×133初始identity差0，实钩Conv MAC12468224与解析相符；参数16180/16180/16365。固定/学习初始全窗差0，三组实际保存/重载输出差0；spatial过去帧梯度0、固定/学习过去帧梯度均非零，全部活动梯度存在/有限。所有8个结果文件逐文件提交，原空stdout因通用*.log忽略而首次git add拒绝，之后显式-f单独保留，不影响科学任务。无正式权重继承，也没有DEV或方法结论。

## 13:04与13:07：两次连续GPU准入都在child前拒绝

首次预绑定记录cuda-qualification001-prebind：11×2秒真实采样，GPU0 meanutil83.4545%、≥90%样本45.45%，显存最低空余27074MiB；虽然超过13312MiB保留要求，算力门槛失败。同期组0–3平均约73–83%，均不合格。parent返回75、没有child PID/没有CUDA资格目录/0GPU更新。

第二次预绑定记录cuda-qualification001-prebind002：新11样本选择同组GPU2，13:07:42结束，meanutil62%、≥90%样本27.27%，空余26686MiB；仍未过≤60%/≤25%要求。parent返回75，无child，不把两次准入拒绝说成两次CUDA资格科学失败，也不改门槛。下一可新采样，不能根据单次低util启动；这不是长期目标阻塞，也不需要改变其他用户任务。

间隙完成官方220输出JOD保存行级资源续接、完整280条件/56均值/28主要比较的汇总，以及四固定联系图代码；五辅助源另冻结，三模块编译正常，但均未运行评分/DEV/渲染。质量manifest的parameters字段保留整个训练checkpoint计数；联合表明确对learned preview及其post只部署16180骨干，另存原stored_checkpoint_parameters，避免给未使用门错误计数。主论文84数/PDF020及全部最终像素保持。

## 13:10–13:11：阶段已推送，第三次资源观察仍无child

fetch正常、main对origin/main为36/0；普通SSH push成功到980e0278，随后实际ls-remote核同一完整commit，push001留证。36小提交含上一阶段末尾一份回执和本轮源码/证据/报告，未合并/压缩/强推，后续合并仍no-ff；阶段完成不等于论文完成。

第三次GPU2新11样本于13:11:31结束，meanutil60.1818%、忙18.18%、空余26686MiB；平均仍超过60%，不为这次仅.1818个百分点的差距放宽标准。原parent/session87233返回75，没有child/资格目录/更新。当前全部资格/预绑定工具句柄终态，无活跃V1 GPU/正式训练/评价任务；下一新鲜连续采样允许后，原源码不变才启动首次实际CUDA资格。不是“正在训练”或科学负结果，不用旧低util快照替代准入。
