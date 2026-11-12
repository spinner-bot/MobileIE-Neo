# R64 最终结果固定视觉检查

登记：2026-11-13T01:41:02.566729+08:00；作者：Codex  (sqs_A100)。R64原两seed训练仍运行，本文件只补预定最终已有输出的显示，不改其冻结训练/评价/汇总源码、质量门槛或预算。最终完整140质量行前不执行，不评价/挑选中间权重，不打开新开发或最终来源。

只在原两个最终CPU runtime和完整summary正常后，用既有输出缓存生成全部10条件×2seed=20联系表。每条件固定首帧0、中帧T//2、末帧T−1，即SYN[0,32,63]、ND[0,8,15]，不由PSNR/误差/外观选帧。每张九列：Input、GT、plain raw/recursive .5、spatial raw/recursive .5、causal raw/recursive .5、causal same-weight current raw。所有组/条件/seed保留，原亮度显示，不调曝光、gamma、颜色或挑局部。

四种CNN raw均只从已完成prediction_manifest对应的owned cache读取/核来源与形状，不重新前向。评价器未保存递归数组，所以仅用每模型已存raw重做同一half-FB/recursive .5显示；保留与原同接受域/首帧fallback/浮点输出，不将这部分CPU重算当额外CNN或新质量试验。显示时才clip/rint到uint8，所有指标继续用原浮点结果。记录来源清单、显示帧号、0新CNN/SGD/GPU/像素来源及正常runtime。

生成manifest的actual_reviewed标记为false；实际view_image后另外提交视觉记录，不以程序生成冒称已查看，不以三时刻静态表当完整播放/MOS/真实感知闪烁。旧R44/R45/R48完整量化强参考仍由原汇总保留，不因为只画同组三组就隐藏强对照。R64 raw对强post与同权重current门槛不变；视觉失败也完整记录，不回写质量选择或自动追加训练。

若最终训练/CPU/summary失败，只保留失败，不执行本显示或重启原任务。20个小型PNG/来源/核查说明分别小步提交；原大数组仍仅zrh项目data，终态阶段再推送，移动演示不改。
