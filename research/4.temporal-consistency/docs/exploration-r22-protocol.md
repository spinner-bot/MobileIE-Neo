# R22 FST与重采样的交换误差诊断

创建：2026-11-11T22:29:00+08:00；作者：Codex  (sqs_A100)；原因：R20增强/warp顺序不等价，GPU共享繁忙时深化原模型算子分析，不扩展任意模块组合。

固定二次FST q(u)=a*u²+b与权重和为1的双线性gather W，有 Wq(u)-q(Wu)=a*[W(u²)-(Wu)²]。这是Jensen/二阶矩的直接代数，不是新方法。差异来自插值邻域变化，不可称时间噪声方差、真实运动误差或全网络差异的FST因果占比；卷积、PReLU、全局门控也不交换。

CPU执行，官方原权重和R14 native plain双seed4000，共3固定权重；既有4个native开发缓存固定frame7→8，共12权重/片段组合。只读已探索数据，核验缓存及权重SHA，不读最终保留素材。对head/body两处pre-FST激活各做同half flow/mask的feature gather，拒绝像素均回退当前feature。使用OpenCV同INTER_LINEAR/BORDER_REPLICATE，FP64核验代数恒等式，再将两种stage状态转回FP32续行原网络。

先用随机小张量/小数平移检验恒等式和整数平移零差；预设FP64scaled residual<=1e-10。实际每组合/层记录valid比例、a、插值邻域二阶中心矩、局部交换差RMS及续行RGB差（raw/clipped MAE255）。raw网络续行必须与正常CPU forward在1e-6归一化输出内相符，不放宽失败门槛。只统计12×2=24诊断行，不训练、不依据结果挑帧、不进行GT质量排名。

该诊断只能判断局部代数是否数值成立、扰动能否传播到RGB，不能证明FST导致R20质量差、提出有效视频模型或证明创新。feature状态12通道明显贵于RGB状态；两个矩的warp也不比直接warp后FST便宜，不作省时声明。后续只有在R21完整控制结果支持时再选择结构研究方向。
