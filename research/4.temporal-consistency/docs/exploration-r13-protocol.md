# R13 训练首帧历史分布与前置上下文

创建：2026-11-11T19:06:47.682456+08:00；作者：Codex  (sqs_A100)；原因：R11同权重实际历史增益仍小，核验pair训练50% self历史是否掩盖可学习窗口；R12已完成但尚在汇总，不按其某个sigma挑本轮模型。

## 假设与唯一改变

R11每两个任务帧中，第一个强制自身历史；实际流仅序列边界如此。保持R5/R11逐步当前输入、GT、两seed、1000更新、4975参数、全模型Adam1e-3/global clip1/Charbonnier .001不变，仅补第一任务帧的前置原始历史。

实拍pair当前索引[t−1,t]，first历史取t−2（t=1时仍首帧自身），second历史仍t−1。合成pair沿原crop/offset恒速外推一帧过去：crop位置减offset，brightness/photons/JPEG相同，额外退化rng=1401300000+seed*1000000+step；不得改变原两当前帧的退化RNG或标签。原margin4足以保证反向±4crop在原图内。独立检查当前/GT/metadata与旧synthetic_pair精确一致。

raw concat_context与teacher-half-FB concat_aligned_context各两seed1000，共4000新时域更新；对齐规则与R11相同，仅现在处理两任务帧的各自历史，不用GT/已知flow。显式统计self边界占比，不能宣称新分布精确等于部署，更不能把burn-in惯例当新颖性。

## 空间不变量和预算

spatial_copy对任何历史输入函数不变，仍使用R11 copy_high两seed作为空间参考。新训练器额外重放copy seed0完整1000步，预设每步task/global-gradient norm/aux norm与R11差≤1e-7、最终model state最大差≤1e-7；保留实际值。若不通过，不默认复用空间参考，先定位，不放宽容差。新训练总计5run/5000更新，实际5run×10周期checkpoint=50份；配置明确4时域+1重放。

时域当前/GT身份逐条匹配R5，初始输出及未clip主干梯度差与plain容差仍2e-6/2e-5，首帧自身回退继续保留。空间模型初始函数及完整重放确保额外历史生成没有意外改变训练数据或优化器。

## 评价及判断

两个新时域×2seed×causal/current×input-reset开关×raw/half01×同46条件=1472新行；复用R11完整2208参考，不换素材、当前监督、流或hyperparameters。预期若上下文分布有用，同权重真实历史相对current以及相对R11对应历史的增量应增加；若只改善current，仍归为训练变化而非部署历史优势。画质/SSIM/edge/时间同时报告、分源聚合，不取单域PSNR赢家。

46条件中的native与resize共享4dev物理片段，旧test已反复探索，无独立最终测试声明。GPU计划同组2/3，逐任务两次空闲预检，最多4空闲卡；算法预算固定、不设墙钟截止、失败不覆盖。代码、检查、每文件结果、过程和报告分别密集提交。
