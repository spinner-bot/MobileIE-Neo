# 真实夜间来源与RAW/ISP评价边界

记录：2026-11-12；作者：Codex  (sqs_A100)。本次定向阅读资料获取/预处理，未复现作者训练，也没有运行新夜间像素评价。

实际阅读[DRV作者论文](https://vladlen.info/papers/DRV.pdf) §3、§4及§4.1–4.2，与[官方源码](https://github.com/cchen156/Seeing-Motion-in-the-Dark/tree/e3e81e742489f997bba30bf331c8d3fa58ddc2b0) README、train.py、test_video.py、三个split文件互证。10页PDF及文本保存在自有cache，获取摘要见process/external-drv/paper-acquisition.json。没有声称逐页精读或复现全文。

DRV用Sony RX100 VI记录3672×5496 Bayer RAW连续拍摄，论文约16–18 FPS、缓冲约110张；演示输出20 FPS不能当所有采集时间戳。202个静态场景有长曝光参考，官方场景级train129/val24/test49互斥；动态Mxxxx序列没有可用于定量PSNR的对应长曝光运动GT。论文低照范围主要约0.5–5 lux，增益约120–300。读取同一长曝光参考做曝光标定属于有参考协议，不能迁移成无参考手机部署假设。

官方VBM4D_rawRGB不是未经处理的低光sRGB：先黑电平/Bayer处理，两个绿色平均形成三通道，2×2 binning及曝光增益/裁剪，再做时域VBM4D，之后CNN映射到sRGB。代码读取16bit PNG并除65535，OpenCV BGR与VGG/RGB次序还需对应。虽然后续CNN可以逐帧，整个既定输入处理已经使用多帧；不能把它列作原始单帧输入，也不能省略时域预处理成本。

作者训练针对RAW到sRGB，整体图像训练对白平衡重要，使用VGG感知与自一致约束（权重.05）、1000 epoch Adam，前后半程1e-4/1e-5。这与本项目低光sRGB到GT的短预算MSE适配差异很大。将增益后RAW直接压到8bit再交给MobileIE会额外引入量化/ISP选择，必须先登记并验证黑电平、曝光、色彩、动态范围与时间处理，不能把渲染差异当模型闪烁或跨域优劣。

六个官方GCS ZIP的central directory已取得，总读取1,823,497字节，仅元数据、0RAW/PNG成员读取。第一次解析失败和第二次标签误配均保留。最终以process/external-drv/classification001/catalog.json及其索引为准：严格按目录识别静态四位ID或动态Mxxxx，不能按0001_*.ARW文件名认成静态0001。202静态场景在short与long齐备，目录实际只见22个动态场景，而test_video.py枚举23个；不能宣称完整23动态数据已到手。四个未来train候选0001/0072/0138/0202仅按预定排序产生，没有看图选片，也尚未下载它们的像素。

另读[Night Wenzhou论文v5](https://arxiv.org/abs/2212.10772v5) III-D/Table IV，核查[作者库](https://github.com/ShenZheng2000/LLIE_Survey/tree/98d778d5b6686ba2d4318f566ef9d7e3f6403551) README及全部issues1–7。论文描述GoPro HERO7 Silver街景1440×1920和DJI Mini 2航拍1530×2720，声称30 FPS、总2小时3分/26.144GB，是真实夜间、未配对资料。原Drive完整链接匿名404。issue1只谈指标，不能由此推断没有备用入口；[作者issue6回复](https://github.com/ShenZheng2000/LLIE_Survey/issues/6#issuecomment-2597100240)给出部分帧/视频备份，issue7尚未恢复完整集合。

已登记并启动作者公开备份Wenzhou Night.rar的正常匿名下载（1,170,848,021字节），资产在zrh项目data。当前下载尚未完成；之后只列归档目录，确认哪些是原输入、哪些是作者增强输出。不能把部分备份称完整基准，不能把增强成片当低光输入。此阶段没有解码、选取评价视频、GT或CNN结果；无参考亮度/自然度与时序指标后续须另定协议，不能用PSNR表替代。

研究上的直接作用是补齐可靠的实拍来源与输入定义，避免用合成DEV结果冒充夜间泛化。本文不是新算法证据，尚不解决创新点、独立最终验证或手机实测缺口。
