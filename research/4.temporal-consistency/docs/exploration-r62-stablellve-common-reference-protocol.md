# R62 已有StableLLVE公开clean权重的共同CPU参考

推理前登记：2026-11-13T00:08:44.106910+08:00；作者：Codex  (sqs_A100)。原R15已完成官方权重/代码/入口与184逻辑行，并非新取得模型。本轮只补现行10源/CPU共同指标和经典recursive .5，填补当前论文参考表；不是方法创新、完整噪声训练复现或模型选择。

固定R15 manifest的zkawfanx/StableLLVE commit4cc070da328ffadc8498285985aa98175e473bc5，公开synthetic-clean checkpoint，SHA277e4d29566709a31f8f25c401396e9bd9c70395e6da36513cd9d9a83e005ad9。4316259参数，U-Net推理逐帧/未来0，保留其BGR、float64除255后转FP32及内部奇数尺寸处理，不加resize/pad/去噪/亮度/GT mean修正。已存在原入口smoke资格，不重做优化器/位级检查。

固定R44/R53/R56共同资料：R30六SYN开发输入各64帧与R10四native_crop ND输入各16帧，共448；原GT/reference flow/mask不变，不取Night/DRV/手机/最终保留素材。每源一次CNN生成raw，再用原自身raw half-FB/warp recursive .5、首帧raw、无reset/GT/调参。10 CNN序列/448中心、20质量序列/896逐帧；唯一观察448，两个输出不是两个独立模型种子。

CPU2/OpenCV1、FP32、同当前exploration_r2_data.evaluate和固定GT参考接受域；保存全部逐帧/逐源/预测来源、输出数组、source/weight与正常终态，保留失败。计时仅CPU神经阶段，不是完整流水/手机时延。没有训练/GPU、新权重/seed或alpha/loss/训练时长搜索；R60活跃源完全不修改，也不将本轮附加到其注册96行。

公开clean权重与已适配8k模型的训练条件不同，必须单独标注project updates0、原clean任务与噪声域迁移限制，不借结果建立公平架构/SOTA排名。只能作已发表时间训练增强器的原权重迁移参考，不称重新训练了StableLLVE全部论文方法。

完整20行后才汇总同两域表；若写入论文，生成器/来源清单、说明、PDF编译和实际版面检查另作阶段，不能把未完成输出或旧GPU不同资料的SYN数值拼进共同CPU表。当前13页build007保持到新完整结果充分核验。

所有新数值资产仍自有zrh data，旧权重不下载/借环境。协议、代码、数据结果、过程/报告分别小步提交，阶段fetch核分歧后推送，no-ff与指定作者保持。论文方法目标仍active，不以补参考替代创新。
