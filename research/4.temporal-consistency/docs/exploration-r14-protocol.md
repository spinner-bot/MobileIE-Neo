# R14 原生区域与缩小图的成对深度训练

创建：2026-11-11T19:22:57.661019+08:00；作者：Codex  (sqs_A100)；原因：R12证实原生多帧窗口，R13减少self训练历史未形成主要增益；转向训练数据尺度/上下文与优化深度，不继续加模块。

## 因素、算法预算及参考

保持同12训练片段、同GT配对、同真实scene/t抽样、同Kodak合成流、全模型Adam1e-3/wd0/global clip1/Charbonnier .001；比较resized（整帧AREA320180）与native_crop（原生320×180同坐标区域）。每种制备训练plain4075、copy4975、aligned_concat4975，各seed0/1，固定累计step1000和4000。全部候选/两checkpoint均报告，不依dev选最佳步数。

native六run从官方权重开始各4000=24000新更新。resized六run从已存1000的模型与完整Adam状态继续到4000，各3000=18000新更新；共42000新更新。plain续R9full_lr1e3、copy续R11copy_high、aligned续R13concat_aligned_context；固定SHA、严格model/optimizer形状/step1000/学习率状态校验。原训练器不支持自动resume，本轮独立实现显式1000→4000延续，不能重置Adam后冒称同预算轨迹。失败另记，不覆盖。

native与resized在1000有同预算比较，在4000也有同累计预算比较。resize1000直接复用原六权重；native与resize4000使用本轮结果。继续保留R11旧aligned及FastDVDnet等强参考，不能只选较弱的R13结果宣布优越。

## 数据制备与上下文

先核验原384训练PNG文件SHA，CPU保留uint8原生数据；只读本来12训练片段，不加dev训练。每step native固定crop RNG=1401400000+seed*1000000+step，x均匀0…1600、y0…900，width320/height180。当前两帧、GT和所有前置历史用同ROI。resized沿原uint8 AREA后转FP32流程，不加量化或新噪声。原生与缩小改变噪声、内容/FOV、运动像素及全局池化上下文，是联合训练分布对照，不能当纯噪声因果干预。

第一实拍任务历史取t−2，边界t=1自身；second取t−1。合成当前/GT与R5一致，first独立退化历史沿R13反向offset、rng1401300000+seed*1000000+step。aligned沿R13冻结官方teacher batch1、半尺度FB；拒绝处填当前，不用GT/已知运动。native ROI通常不同，不能把不同crop误作同一缓存key；可仅在同一步两个邻接任务安全复用。

随机scene/t前1000逐条匹配旧记录，后3000保留相同抽样定义。样本metadata记录ROI、sourceSHA、context和flow诊断；保持任务当前/GT在同一制备下跨模型完全一致。FP32、cudnn deterministic，不设墙钟截止，仅固定4000轮数。

## 评价预算与限制

同46条件，plain/copy各1历史、aligned causal/current，共4调用×2seed×2train制备×2step×46×2input-reset×raw/half01 =5888配置—序列行。resized1000的1472行复用，新增4416行；相同无事件或静态reset可复用完全相同输出/指标，但须记录实际计算与alias，不能称5888独立视频运行。新dev/native仍同4物理片段，旧test已反复探索，未用官方test和Kodak13–24不读取。

保留PSNR/SSIM/edge/GT残差时间、逐帧事件及同权重current-history；分别评价尺度效应（固定step）、预算效应（固定制备）与真实历史贡献。最终仍需自然夜景、完整强LLVE训练与端到端/手机成本，当前不作创新或最终性能声明。

## 资源与留档

/data ext4挂载已复核，原生CPU像素约2.4GB/进程，当前可用内存约722GiB、数据盘余量5.7T；权重/像素留忽略cache，Git只保留清单/SHA及结果。计划GPU2 plain/copy、GPU3 aligned，仅同组，启动前各自两次空闲预检；最多4空闲卡，不抢占别人进程。本轮较长训练无墙钟截断，按周期1000保存，持续小步提交、记录过程及阶段报告。
