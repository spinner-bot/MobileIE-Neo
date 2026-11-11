# R8：从原始双帧输入学习关系的根部强基线

创建：2026-11-11T17:54:06.044108+08:00；作者：Codex  (sqs_A100)；原因：R7冻结特征融合在实拍退化，给定相同motion仍未超过普通RGB融合，需补齐端到端双帧输入强参考，而非继续拼接特征模块。

## 问题与模型

保持原head/body FST、注意力与tail。只在head首个Conv5输出、PReLU之前加一个无bias的3→12 Conv5分支，初始化全0（新增900参数，共4975）。主干来自相同官方slim，初始函数与原模型一致。五种输入信号：

- spatial_copy：current，同形同学习率分支，但两Conv5可合并为原单帧Conv5；控制更快首层优化带来的假结构收益，无新增空间表达能力。
- spatial_detail：current−3×3均值(current)，单帧强空间容量/去噪对照；边界均值不计补零。
- temporal_concat：previous raw RGB；序列首帧用current。是基本双帧卷积参考，不宣称新颖。
- temporal_delta：previous−current；首帧为0；零历史差自动恢复当前主干。
- temporal_delta_aug：同delta，但训练时仅给每个pair的第二帧历史做10%替换为current、10%替换为另一pair的前帧。原始四帧输入和目标保持，避免把错误历史当当前GT。

concat与delta本质上可重参数化等价：A*current+B*previous=(A+B)*current+B*(previous−current)。它们的初始/优化路径不同，但不能把差分形式当新表达能力。current替代时concat的两Conv5可加权合并为单帧Conv5；delta/current则恰好是其训练后主干，不扩大空间表达能力。spatial_detail提供真实额外空间上下文容量控制。

## 固定训练

沿R5完全相同real12_mix、种子0/1、1000更新，五新模型×两种子=10次/10000更新；固定最终1000步，不测试选优，无墙钟截止。复用R5 plain两种子1000步作为同数据/预算主干基线，避免重复训练；逐步校验所有新模型的当前输入/GT样本身份与R5。

Adam主干lr1e-5、辅助Conv5 lr1e-3，各自梯度范数裁剪1，weight_decay0。分支从0学习，相同LR/容量控制均公开；这不是官方长训练复现。损失沿R5任务Charbonnier，不新增时序损失，避免同时改变多个因素。原始主干参数列表保持原顺序，新增参数独立裁剪，不稀释主干学习率。

aug独立RNG=1208000+seed*1000000+step；两个pair依次draw uniform。<.1给第二帧self历史，[.1,.2)给other_pair_previous，其余actual_previous。首帧监督不改，current/target张量绝不原位修改。所有选择写入trace。其目的为检验事件鲁棒性，已有历史drop/错配训练先例，非新技巧声明。

## 验证与评价

先核验初始输出/主干梯度与plain一致、辅助梯度非零、pair与batch1 stream对应、RGB状态独立存储、current-history折叠恒等式。输入历史是原始RGB，有限上一帧，无未来和额外历史特征前向。

评价沿R5的38探索条件，raw/共享half01、无reset/原R2输入reset；所有输出融合也用同一reset。两个spatial和plain仅causal一种（实际单帧）；三个temporal各causal/current两种。两种子总2736评价行：两个新spatial共608、三temporal共1824、复用plain304。记录逐帧事件、同权重历史替代、重置和域间取舍。共享motion/teacher非免费，不能把缓存质量测量当端到端速度。

只在存在有意义的历史独立收益后进一步测完整模型延迟/实际状态和新数据迁移；无收益则保留否定结果、收缩该具体方向。当前材料仍探索，不是最终独立测试或论文创新证据。

## 资源与过程

训练计划同组GPU2/3两项新guard；需要时可使用同组其他空闲卡，总数不超过4，资源变化保存拒绝并换新记录。代码、每次训练trace/checkpoint哈希、评价、过程和报告分别小步提交。移动演示支线不介入。

修改：2026-11-11T17:58:47.598966+08:00；作者：Codex  (sqs_A100)；原因：GPU训练尚未启动，补齐spatial_copy优化控制，避免auxiliary较高学习率的静态收益误算视频容量；增加两次1000更新，预算及评价计数已相应改明。
