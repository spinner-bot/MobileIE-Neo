# R26 监督目标与聚合映射的强控制

创建：2026-11-11T23:33:11.146890+08:00；作者：Codex  (sqs_A100)；原因：R24单节点及R25GT估计对应都未统一恢复主要PSNR差，转向训练目标，避免继续无目标加模块。

三种4075参数映射：plain使用R14原CausalFST/plain；pre_fixedhalf/post_fixedhalf沿用R21，固定alpha.5。两目标分别为原Charbonnier sqrt(error²+1e-6).mean和原尺度MSE error².mean；均作用于同四任务完整raw神经输出，不先clamp。无新增时序/颜色/辅助损失，无可学alpha、schedule或结构参数。

三映射×两loss×双seed×4000=48000新优化更新，从官方预训练重置；每seed六独立网络/Adam按step同步，只共享一次固定teacher的历史对应。全部当前/GT/ROI/合成上下文与R14原生4000项精确匹配，历史SHA与旧R14/R21对照只记录，不假定跨GPU自动相同。Adam1e-3/wd0/global clip1保持，无墙钟截断、按算法4000步完成。MSE不根据dev或观测梯度重新缩放，原尺度损失会改变梯度幅度/clip和Adam耦合，故这是完整目标/优化设定对照，不是只改变梯度方向的纯因果实验。

每步记录实际目标、Charb/MSE双诊断、全局RGB/中心化MSE分解、global norm与clip是否触发、共享上下文引用。首步同映射Charb/MSE初始输出应完全一致，记录初始未clip梯度范数比与方向cosine，不用其调整超参；current替代仍与原空间函数保持预设<=2e-5归一化最大差。全部网络/Adam指针不共享、teacher不求梯度。

每1000步保存模型/完整Adam/trace前缀SHA。Charb组在固定检查点逐tensor比较旧R14 plain及R21 pre/post的模型/Adam；记录真实是否完全重放，不以文件SHA（含不同config序列化）推断数值不同，也不因不精确而放宽静态函数门槛。旧步骤task差另记录。所有新网络都完整执行4000，不以最优验证选终点。

评价固定1000/4000、原46探索条件、所有seed；同一轮全候选/参考使用共同后端，优先空闲预检后的CUDA，若资源不足则预先记录CPU后备并完整重算，不能混旧CPU/GPU数字。plain每checkpoint报告raw、原recursive half01、finite postwarp.1/.5及两reset；pre/post报告actual/current、raw/half01及两reset。共8832逻辑行：plain2944、pre/post5888。新的MSEplain加经典后处理是必要强基线，不能只和旧Charb后处理比。共享输出计算不当独立方法延迟。

GPU仅在新两次idle预检后使用同组1/3等最多两空闲卡，每job一个seed六模型；若被占则继续等待/CPU独立工作，不挤占别人。GPU/环境不改全局设置，作者/main/no-ff及逐文件提交保持。完整代码、检查、过程、结果、报告分开提交并阶段推送。该轮为目标/优化强对照，MSE或平均都不作为新颖性；据结果再推进少数统计适配结构。最终保留test/Kodak13–24未读，论文目标active。
