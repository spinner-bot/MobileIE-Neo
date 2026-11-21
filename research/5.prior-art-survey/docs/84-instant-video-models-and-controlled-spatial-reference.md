# 84｜Instant Video Models与受控空间稳定化参照

2026-11-21T14:58:48.651036+08:00；Codex  (sqs_A100)。本回定向读取[NeurIPS2025正式入口](https://proceedings.neurips.cc/paper_files/paper/2025/hash/a12779b5e802668df1cbc73fa00da62f-Abstract-Conference.html)、[v1原文](https://arxiv.org/html/2512.03014v1)§3–5及D/E选择段。其因果适配器可稳定中间特征与输出，控制器使用当前/过去输入和状态；空间版本学习局部历史核。论文分析范数距离下的准确性—稳定性损失，并有增强任务。一般冻结适配、学习衰减/空间融合均非本项目创新。未读全篇/原图或复现作者成绩。

[作者代码](https://github.com/wision-lab/instant-video-models/tree/e685f56316ae1476fa8831c8c60b33bc5e179af7)与MIT许可已核，选定源原样入third_party。代码的3×3融合为9个过去logit加固定0的当前logit；采用作者−4偏置，不能直接把当前总权重说成.98。实际RMSE含epsilon，范数理想结论不自动覆盖它或本项目旧MSE。作者README亦说明清理后未重测全部配置。

本项目判断：R37/R64/R73/O1/L1等局部失败不等于该已发表方法已经复现。它的局部多位置分布与先压成单点再warp有不同限制，但普通空间核也已有先例；这里补正式基线而非为旧模型换名。下一I1将原32/64控制器及完整Conv2d/输出hook用于冻结F0，配对无空间版本；尚需资格/正式学习/整链成本，不能叫4K或手机实时。任务/数据/训练来源不同，须写适配而非作者原成绩极限。全部旧STOP与最终封存保持。[精确读取范围](../process/instant-video-i1/source-reading-pointer.json)。
