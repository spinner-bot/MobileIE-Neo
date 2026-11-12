# R58：四个固定DRV静态训练源的RAW小样本与确定性渲染

事前登记：2026-11-12T22:13:00+08:00；作者：Codex  (sqs_A100)；原因：R53/R56参考和R57输出类上限已完整收口，推进真实夜间资料，不追加局部模型网格。

## 固定成员与阶段

沿R52像素读取前已定的官方train排序四候选0001/0072/0138/0202。权威索引classification001；每源只取按文件名排序最前连续8张短曝光ARW、该源唯一长曝光ARW，共32+4=36成员。不取任何PNG、VBM4D_rawRGB、val/test或动态Mxxxx成员。只能从官方GCS short1–4.zip/long.zip公开Range取得这些成员，验证206/Content-Range/原ETag、ZIP头、大小与CRC。大体量原件及临时压缩流只进zrh自有data，完整来源/CRC/SHA与失败在Git清单。最多4个HTTP工作线程；socket错误可对同一固定Range最多3次技术重试，不更换数据/端点或绕过认证，不重启其它下载。

先取得36个完整成员并记录正常终态，再读取EXIF/RAW和渲染。本阶段不自动跑CNN、训练、时间指标或挑参考。渲染前检查所需元数据及资格，任何失败保留而不默默换帧/场景/规则。R52六归档目录事实不改；Night原3514885/3531976下载→目录链继续，不改核心或从partial RAR解码。

## 环境与确定性渲染

Python3.10项目自有venv仅新增官方PyPI rawpy0.25.1与ExifRead3.5.1，固定兼容wheel的SHA；不升级numpy/torch/Python或借环境。最新rawpy0.28需Python≥3.11，不据此升级项目。原rawpy0.25.1源码/官方Params文档与实际安装参数互证；用户缓存、pip和临时文件仍走project-python。

每源用第一张短曝光RAW的camera_whitebalance作为固定user_wb，所有8短帧及对应长曝光采用同组WB；不从GT像素或长曝光WB选颜色。EXIF读取ExposureTime、FNumber、ISOSpeedRatings，要求正数、同源长短ISO和光圈相同；否则该源渲染停止，登记不支持原因，不以GT均值补偿。按EXIF长/短曝光秒比给短帧线性RGB缩放；长曝光gain=1，不按输出图调增益。相机型号/可见尺寸/色矩阵和朝向应匹配；原黑电平/白电平由LibRaw元数据处理并完整记录，不硬编码别的相机值。

rawpy统一half_size=True、user_flip=0、固定user_wb、use_camera_wb=False/use_auto_wb=False、no_auto_bright=True、adjust_maximum_thr=0、bright=1、gamma=(1,1)、output_bps=16、output_color=sRGB、HighlightMode.Clip、FBDD关闭/无wavelet阈值/无median去噪。这里输出为线性sRGB原色RGB16，既非原Bayer也非作者VBM4D输入。预先固定两个有不同任务含义的视图：native只用gain=1，保留短曝光，才是原亮度增强压力输入；ev_matched使用EXIF曝光比，属于已匹配目标亮度的极低光去噪诊断，不能直接作为LLIE输入排名。长曝光GT两视图共用gain=1。随后在float64中乘各自已定gain、clip[0,1]；线性域全FOV INTER_AREA到长边640且保持比例，再施加标准sRGB分段编码，最后round/clip到uint8。两视图全部保留，不按结果选视图/调曝光比例。原件保留，记录全部版本/参数、尺寸、元数据及裁截比例；16bit渲染引入的量化也须披露。不得逐帧自动曝光/白平衡或GT-Mean。

先无真实像素的sRGB编码黑白端点/分段连续和单调资格；实际RAW渲染还须有finite、形状、固定WB/无自动亮度的配置资格。每源保留8帧输入+单GT的自有缓存、联系表与像素来源清单，实际查看后另写视觉记录。若长短曝光元数据/颜色链不满足规则，不产生伪配对评价；修改需另记原因/协议，不能回写结果后假称事前。

## 判读边界

这四源属于官方train、静态短片且只有8帧，是新获取的探索资料，不是完整标准/独立最终动态夜间测试或手机实测。原采集约16–18FPS，不能硬写30FPS或假定固定时间间隔；EXIF时间若无子秒就保留未知。长曝光只是静态参考，不给动态序列算PSNR；场景/灯光变化和微小错位还须检查。

本项目渲染不同于作者Bayer拆分/binning/VBM4D→ResUnet，不能用作者网络成绩作为同链基线，也不能称复现其方法。后续模型评价另立固定协议，所有适配前权重/正负结果公开，不按这32帧选训练点或修改渲染。新资料获取与可信ISP只补外部有效性基础，创新机制及完整论文目标仍未完成。方案、源码、依赖、获取/渲染结果、过程与报告分别密集提交，阶段推送；不改移动演示支线。

修订（任何RAW成员获取/解码前）：2026-11-12T22:16:00+08:00；作者：Codex  (sqs_A100)；原因：完整EV补偿已把增强问题改变为去噪，不能将其域差误作LLIE能力；因此固定保留native与ev_matched两个明确视图。未改变36成员、场景/时间选择或增加训练/算法搜索。
