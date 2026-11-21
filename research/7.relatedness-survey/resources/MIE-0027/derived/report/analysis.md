# MIE-0027｜CBGAN：低分辨率变换表示的相关先例，正文与实现证据受限

2026-11-21；Codex  (sqs_A100) <langxibielangle@qq.com>。有效序30。当前为证据受限分析：0页PDF、0张科学图表取得，不能称全文或代码审查完成。

CBGAN 是需要优先补证的直接相关工作。它的书目准确引用MobileIE，摘要提出低分辨率双边空间中的变换预测与全分辨率应用，可能影响二期的表示及成本取舍。当前证据不足以判断具体继承关系、时序设计、原实验优势或方法重复程度。

## 来源与已确认事实

[出版方页面](https://www.sciencedirect.com/science/article/pii/S2667379726000975)对应Biomimetic Intelligence and Robotics，100369，五作者Chaoyi Liu等；官方API给在线2026-08-31、CC-BY4.0，索引页面显示InPress/JournalPre-proof。许可与身份已核，实际正文尚未取得，不能把注册元数据的version字段当已看正式排版PDF。

[Crossref登记](https://api.crossref.org/works/10.1016%2Fj.birob.2026.100369)的51参考中，b17含MobileIE完整作者/标题/ICCV页码和DOI。它确认直接引用，但正文语境、印刷序号、是否继承骨干或loss均未知。见[引用溯源](../docs/citation-provenance.json)、[来源清单](../../sources/manifest.json)。

作者摘要称方法为Cycle Bilateral Generative Adversarial Networks：低分辨率流预测双边空间中的局部/全局仿射变换，在全分辨率应用；采用对抗训练、逆映射及cycle consistency。它报告4K照片几十毫秒、相对ResNet生成器CycleGAN约少95.8%参数、超过170倍速度及无监督质量优势。这些只是摘要归因；无绝对参数、精确时间、实验表或独立验证。

## 证据边界与获取过程

原200响应仅2683B跳转页；新1966B API XML只有coredata；两次403大HTML经语义检查仅为内容未提供页面，字节数相同而SHA不同，均无论文摘要。它们都不算正文。公开下载403、API并发429后停止并发，191.6秒冷却后的单次FULL请求401 InvalidAPIKey，PDF格式406；没有借用凭据或继续循环。正常浏览器仍403，无PDF点击/下载，已关闭；实际查看的是访问诊断截图，非科学图。

替代学术索引仅给DOI开放位置，作者主页及精确题名/模型/作者/GitHub查询未找到可核的正文或官方实现。没有发现明确而未履行的代码公开承诺，也不能据检索结果证明私人/异题资料不存在。全部13普通回执（含初始一）及浏览器记录留存；详见[过程笔记](../docs/reading-notes.md)、[访问诊断](../docs/acquisition-progress.json)、[HTTP语义复核](../docs/http-payload-semantic-review.json)。

缺少grid尺寸/guide/系数和采样、模型结构/参数范围、判别器与目标域、loss权重、训练数据ID/许可/外部先验/预算、完整质量/消融及手机工件与成本协议。无配对监督不能直接解释为零训练、无需目标域或无需高质量数据。机器人检测/分割/SLAM在摘要属于动机，不能补成已验证的下游结果。

没有精确分辨率、设备、precision、后端、batch、计时边界及内存/热功耗，4K几十毫秒不构成手机或持续视频FPS。95.8%是相对一个baseline的比例，不是MobileIE四千参数规模；170倍亦非相对MobileIE的速度。历史/未来帧、state、时间loss/指标和视频评价未知；书目FlowNet2不能证明采用流，摘要未提时间也不能证明没有时间实验。

## 二期研究价值和待证问题

值得保留的方向是低分辨率变换预测与全分辨率执行之间的表示/带宽取舍，以及相应guide、系数和边界在运动/曝光/噪声下的敏感性。它们尚非已成立的二期方法。双边grid、仿射变换和cycleGAN有前驱，不能改名申报创新。

普通基础书目与检索浮出的2007双边grid视频吞吐描述已按冻结范围处置：当前没有新的MobileIE特定实现或时序一致性具体联系，不因关键词video无限扩展；见[线索处置](../docs/lead-dispositions.json)。若新正文/代码给出具体合格联系，须先做Gate1增补，不能忽略。

后续控制包括同目标域/数据/预算的无监督对照、grid/guide/fullres采样的完整计算与内存、真实动态条件下空间—时间/感知联合评价，以及相同精度/后端的端侧全链成本。先合法取得正文/图表/确切实现与配置，保留当前终态并追加复审，再由全资源综合筛选下一实验分支Gate3。当前不执行预实验。

[14项claims](claims.json)区分作者报告、取得证据与未知；结项以[Gate2审查](../docs/gate2-review.md)和[决定](gate2.json)为准。0研究代码执行/模型加载/推理/训练/GPU/Gate3；普通网站JS仅用于访问诊断。
