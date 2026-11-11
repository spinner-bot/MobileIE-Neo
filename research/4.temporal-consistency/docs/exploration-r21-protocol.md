# R21 聚合感知训练与历史初始化控制

创建：2026-11-11T22:05:08.430817+08:00；作者：Codex  (sqs_A100)；原因：R19有可用信息、R20固定前置被强后置支配，分离未适配训练分布和优化起点，暂不增加通用大骨干。

四类均保留原4075参数MobileIE骨干：pre_fixedhalf为F(x+.5*(A-x))；pre_learnzero/pre_learnhalf为F(x+alpha*(A-x))，全局scalar alpha分别0/.5初始化（额外1参数）；post_fixedhalf为.5*(F(x)+F(A))，原始神经输出先聚合，公共输出阶段才clamp。alpha直接参数化，Adam后投影[0,1]，不在forward用饱和sigmoid/clamp断梯度；fixed alpha不加入state buffer。该组是经典聚合/共享映射/初始化强控制，不称新方法、KPN复现或最终创新。

同R14 native12训练目录384PNG、320×180 ROI与全部4000 step当前/GT/合成/前置历史上下文，双seed。用固定官方teacher半尺度FB对齐四个任务历史；每step仅计算一次，四模型使用同一实际historyTensor；模型参数/Adam完全独立且data_ptr无交集，按step同步推进以共享昂贵对应计算，不共享权重/梯度。每step当前监督/ROI/合成上下文与R14 native plain全量逐项一致；另记录与旧R14 aligned历史SHA匹配情况，不先假定跨GPU旧历史逐像素一致。无GT motion部署。

四类×双seed×4000=32000新优化更新，官方预训练初始化，强全局Adam1e-3/wd0/clip1、Charbonnier .001，无新增时序/辅助损失、无KPN退火。alpha梯度参与global clip，记录投影前后数值、梯度、loss和norm，不能忽略优化耦合。每1000保存模型/完整Adam、共享上下文trace与各自训练trace前缀SHA；固定1000/4000全部评价，不按dev选checkpoint，不设墙钟截断。

初始current替代必须保持原单帧网络：预设最大归一化输出差≤2e-5、任务差同记；post两分支共享骨干，重复当前只是静态控制，不代表实际历史贡献。pre_fixedhalf与learnhalf初始因相同alpha应数值匹配，另记；真实过去初始输出允许变化，不伪称与原始current等价。CPU检查含静态极限、nonzero/zero初始化、同骨干参数独立、边界alpha可获得梯度/投影和post原始平均定义。失败留档，不放宽门槛。

评价原46探索条件，四类双seed、1000/4000、真实history/current替代、reset及raw/共同half01，新增5888逻辑行（4类×2seed×2checkpoint×2history×2reset×2fusion×46）。全部模型走同inputteacher对齐，复用既有R11推理和R14强参考；post骨干每输出两次前向、pre一次，额外teacher/flow/状态不作零成本。current替代后若保留整体收益，不能把它归为推理历史利用。最终未读test/Kodak13–24保留。

先做CPU检查再启动每seed独立GPU job（2/3空闲预检后）。一job四模型并列推进各4000，shared teacher不接受梯度；每个checkpoint/过程/报告逐文件小步提交，完成阶段以zrh目录spinner-bot身份SSH推送，保持no-ff。后续如需防参考帧塌缩损失，先核验KPN等已有方法，不把该问题或常见技巧当新颖性。
