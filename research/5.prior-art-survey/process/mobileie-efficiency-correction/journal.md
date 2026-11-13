# 原模型效率引用纠正过程

2026-11-13；Codex  (sqs_A100)。上一回合verified wait，本回合纠正实际发现的早期效率标注，不扩实验库。

1. 核/data挂载及原四PID存活；早期research2 README和HTML把1100FPS标为SoC，与同表6.72ms矛盾，且4047/4075参数口径混用。现有paper reference-budget表为4075及手机ms空值，不需要从旧表移入手机数字。
2. 定向读取作者arXiv2507.01838v1表1/表2与§4.2；表1 .895/6.72ms分别为GPU/SoC、FPS1120.584，与前者量级对应。普通CVF检索定位正式元数据/正确缩短文件名；实际正式PDF403，未绕过或声称正式全文已读。仅以实际v1给数字来源，不将抽象1100FPS直接当手机。
3. 空CUDA mask、自有project-python，只实例化原MobileIELLENetS(12)并遍历parameters(recurse=False)，0forward/已训练权重/视频/SGD/GPU。注册4075、两个FSTS各14，剩余4047；保存原代码和纠正前README/HTML指纹到inspection001。它只说明计数差，不证明历史THOP具体行为；没有重跑历史工具或Windows速度。
4. inspection独立提交后，分别修正README与HTML。保留原1.447ms/691FPS，不把型号差异猜作完整因果；区分原文GFLOPs/旧工具GMAC及完整视频成本，源6.72ms倒数约149只作算术，不当手机实测。HTML改动没有读取或重编码内嵌图。
5. correction-check001核两份原嵌入图base64 payload与前版一致、错误SoC1100标签移除；原17训练源及27/37评价源均未变。四原PID/start同身份活跃，训练最近两seed8700，无runtime终态、新DEV/JOD、重启或额外GPU。结果与note61解释分别提交，原论文018不重建，移动演示research6未改。
6. 报告0120/索引按文件另交，阶段fetch仅main、如有分歧显式no-ff，不rebase/squash/强推。纠正不能代替新方法、外部自然动态、完整手机成本与独立最终；目标active，原训练和既定完整评价继续。
