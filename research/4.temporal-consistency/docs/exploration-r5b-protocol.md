# R5b：排除“缺少时序训练信号”的解释

创建：2026-11-11T16:24:51.832855+08:00；作者：Codex  (sqs_A100)；原因：R5 部分完成的探索评价中，coarse 历史与同权重 current 替代接近；需用有限补充检验训练目标，而非扩张架构数量。

这是根据已经见到的探索结果作出的适应性后续，不能当作独立确认实验。立项时 R5 temporal_post 尚未完成，其他模型主要评价已完成。最终全部 R5 结果仍照原协议保留，不用本轮替换不利结果。

## 唯一变化与预算

选择 plain、temporal_bilinear、tsm_learned 三种，各 seed0/1，重新从相同官方预训练初始化，各1000更新，共6000更新。保留 R5 完全相同的 real12_mix 输入、任务损失、优化器、学习率与梯度裁剪。模型第1000步是唯一主要 checkpoint，无墙钟截止，无测试选优。

新增仅作用于 synthetic pair 的 GT 残差时序 Charbonnier，系数 1。依据是 R2 已有系数1的强时序训练基线，不在本轮验证数据上搜索权重。合成对平移 dx/dy 是生成器已知量，通过整数切片精确对齐，并额外剔除8像素边缘；不估计 flow，不对真实配准残差施加时间项。

对齐误差定义为 current prediction-target 减去 previous prediction-target（previous error detach），均方根式 Charbonnier epsilon=.001。结构内部 pair 历史特征仍可反传；detach previous error 不等于关闭 BPTT。任务损失仍平均2对4帧，时序项只平均1个合成对的有效像素；系数缩放和 R2 全batch时间项不同，须披露。

## 比较与判断

复用 R5 的38条件缓存、raw/half01和历史消融：plain152行，temporal456行，TSM304行，共912行。比较同结构 R5b−R5 和 plain 时序训练对照，区分新增结构与目标优化贡献。逐帧检查 cut/light_step，不只看全片时间误差。

若 temporal_bilinear 仍与 current 替代接近，则降低该粗尺度线性历史分支的优先级；不以训练更久自动掩盖无效历史。TSM 只作为已有机制强参考。混合数据区别时序监督已有 STCL 直接先例，本轮不构成创新声明（见文献调查13）。后续独立测试仍必要。
