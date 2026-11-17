# 71｜共享几何分支、互补描述与可学习对齐：先例先于新核心

2026-11-17；Codex  (sqs_A100)。R87只改变12维已有z的正对角度量而未得可靠匹配；下一候选改从head Conv3之前的12通道A/3×3 patch读取信息，并必须接完整因果对应与RGB状态。没有外部代码/权重/新数据下载或执行，不主张新颖性已建立。

[SuperPoint arXiv1712.07629v4](https://arxiv.org/html/1712.07629)实际读§3共享encoder/两个decoder概述和§3.4已知homography对应监督段落。共享encoder、独立任务输出、合成几何训练描述符已有明确先例；本轮不复现其检测器、完整训练或速度，也不把其homographic adaptation等同本项目的平移监督。

[Beyond Single Descriptors，J.Imaging2026正式正文](https://www.mdpi.com/2313-433x/12/5/201)实际读摘要、§3.3/式4及Figure2文字：同一CNN预测两个描述图，不同损失监督，以两描述向量的内积绝对值约束互补，再组合相似度。不是只有“正交”一词的搜索线索，普通共享/互补描述不能当空白。其逐位置输出向量夹角约束，不等于本候选对固定恢复卷积权重的硬行空间补投影；二者区别不自动构成创新。未运行代码、读全部实验或查看其图像。

[PRISE CVPR2023官方PDF](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_PRISE_Demystifying_Deep_Lucas-Kanade_With_Strongly_Star-Convex_Constraints_for_Multimodel_CVPR_2023_paper.pdf)本次403后停止；只有官方检索正文摘录指向DeepLK/对齐损失景观，不声称读过全文或采用其证明/训练目标。OD-Net只有出版摘要检索线索，不据此描述其正交实现。LK/可学习特征配合已是直接先例，新后端只能称本项目固定稠密Gauss–Newton型局部平移近似，不冒充严格inverse-compositional算法或DeepLK原文复现。

R88具体问题是：冻结原恢复路径，利用较早共享A的额外空间成分，用极小独立描述投影学习已知噪声/几何，是否能在真实因果视频输出上接近两经典参考而减少完整成本。普通投影与恢复W行空间正交补投影用相同初始有效kernel、相同数据/预算/推理形状；受约束训练自由度更少，不能以注册参数相同作严格容量等价。原RGB函数保持来自分支不回写重建，绝不归因于正交本身。W的线性补空间不等于输入图像的真实无影响空间或统计独立噪声。

目标必须是完整视频链：同一次F0前向共享stem、四通道quarter描述、固定多尺度稠密局部平移求解、原生FB接纳、自己的上一fused RGB递归。已知部件、训练方式和投影本身不当创新；完整质量/感知、逐源保护、同权重zero-flow干预、全部状态/匹配/warp/新增conv成本及外部/手机/最终仍需证据。若不通过，不扫描述维度/patch/金字塔/迭代/投影/损失/噪声/LR/长度/seed，不把新分支接回旧STOP追分。
