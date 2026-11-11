# R16 多尺度骨干与同容量历史控制

创建：2026-11-11T20:27:00+08:00；作者：Codex  (sqs_A100)；原因：R14深度/尺度控制仍无足够历史贡献，检验更大空间感受野的时间信息利用。

这是机制探索，非新方法声明。标准encoder-decoder、多帧卷积、原始FST/门控组合均有先例；不把组合本身当创新。原模型高分辨率浅层FST可能难以建模运动/噪声，但该解释尚未证明。

保留官方12通道head、body二次参数、attention/tail；将原body的12→12 Conv3替换为36→12 decoder Conv3。当前head形成高分辨率skip；共享head后stride2 Conv3(12→24)、PReLU24、Conv3(24→24)、PReLU24产生当前/过去半尺度特征；concat48经Conv3(48→24)、PReLU24，bilinear align_corners=False恢复当前尺寸，与12通道skip拼接后进入decoder/FST及原gate/tail。新增decoder24列初始为0，原12列/bias原样复制。其余新增卷积默认PyTorch初始化，seed固定；原官方模型初始化输出预设最大归一化误差≤2e-5，失败保留，不放宽阈值。

三类同架构/容量：spatial（两路均当前）、raw_history（一过去一当前）、aligned_history（固定官方teacher半尺度FB对齐原输入，拒绝区域回退当前）。所有参数均训练。只使用t及t−1，一过去、零未来；边界和reset以当前回退，不循环或窥视未来。推理另做同权重current替代；spatial不重复计算历史模式。参数实测记录，禁止与4k原模型宣称等成本；额外head/low encoder/teacher/flow/状态开销均须报告。

三类×双seed×4000=24000新更新，官方预训练初始化，不从R14最佳checkpoint续接。原R14 native训练与R13前置历史生成器完全复用，12训练源384PNG逐SHA、ROI RNG1401400000+seed*1e6+step、合成监督/历史RNG不变；每步两实拍任务+两合成任务。全模型Adam1e-3/wd0/global clip1、Charbonnier .001，无新时间损失。监督身份/上下文逐步与R14 native trace核对，控制新增参数和空间表征，不重新挑数据。保存1000/2000/3000/4000，固定1000与4000全候选评价，禁止依据dev挑checkpoint。无墙钟截断。

原46探索条件、raw/共同half01、reset、history替代，保留逐帧/逐片。与R14 native plain/copy/aligned及R12已发表模型分别比较，训练任务/未来/参数差异明确。评价alias只复用证明等价项并独立计数。已有dev反复探索，不称最终测试；保留3未读官方test与Kodak13–24。

启动前先CPU验证：初始官方输出/共享权重、同seed空间/时间state完全一致、学习后真实历史确能影响输出及spatial不受历史扰动、奇数尺寸shape/finite。GPU初始化核验在真实任务上重复。先提交源码/协议/检查，再经gpu_guard每任务两次空闲预检，同组最多4卡。checkpoint/过程分小步提交，运行中的日志不当完整结果提交。
