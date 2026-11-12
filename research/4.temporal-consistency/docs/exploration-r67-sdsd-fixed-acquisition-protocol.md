# R67｜恢复官方源后的固定SDSD探索配对资料

登记：2026-11-13T03:37:48.265420+08:00；作者：Codex  (sqs_A100)。官方Drive距前次约31小时后恢复正确206 Range，已有有界来源记录；当前尚无取得/解码的SDSD图像。R65/R66原活跃链不改。

严格承接R17在2026-11-11像素前选定的四个**官方train**来源：indoor pair1/pair69、outdoor pair2/pair80；每侧按帧名自然序前16、严格同名配对，共4×2×16=128PNG。不得按画质/噪声/配准/内容换片，全部25官方测试像素不读。固定官方split commit07330bb0ac124d2d56f6e92908d650dde3f7f7c2与已有Drive分卷清单。它是新的探索DEV，非完整/官方独立测试；ND+提高ISO/机械路径，不称自然夜景/自由动态GT。

阶段1只读取室内/室外末ZIP的EOCD/central directory，每域≤4MiB目录（尾部定位另≤64KiB），不解码/取图像成员。分卷顺序来自作者文件id/name/size，保留ZIP成员disk/local-offset/size/CRC及ZIP64扩展。可构造仅目录的单盘EOCD供Python标准库解目录，必须保留真实volume/offset，不能以该代理打开图像。若目录跨末盘或超界而当前实现不支持，明确失败/另登记，不下载整百GB归档。完整目录只进自有data，选取128成员/来源/SHA/统计入Git。

阶段2只对冻结128成员跨盘Range取local header与payload，最多4个HTTP线程、每请求≤1MiB；同一Range仅网络5xx/timeout等最多3次技术重试，配额/权限/200 HTML不绕过、不重复请求其它端点。每次要求206/精确Content-Range/尺寸；Drive若无ETag如实记，不伪称If-Match保护。原完整ZIP/PNG的官方SHA并未提供；本轮以作者目录CRC/原长与最终解压CRC/大小/本地SHA验证，不能冒称整归档校验。仅ZIP解压、不自动图像解码或CNN；原PNG和压缩块进zrh自有data。

阶段3仅在128成员完整正常后，按R17固定准备：原1920×1080输入/GT必须同形/同名、自然排序连续，原生ROI x704:1216/y396:684及全FOV AREA480×270，两侧规则相同、RGB uint8缓存、无有损/自动曝光/GT mean/颜色修正。没有改用作者960×512 npy。记录原形状/PNG dtype/通道/排序，不假称已核传感器间隔/原采集30fps；如输入不是原8bit RGB或未配对，保留失败，不偷偷丢16bit信息或换帧。

完整8个视图缓存（4来源×2视图），每个16对时间帧；源ROI和缩图是同一资料，不能当8独立视频。可预先固定所有来源首/中/末输入/GT与中心128²细节，生成不等于实际审阅。静图不证明完整配准/无ghosting；GT估计流仍非真实运动。后续flow/增强器质量和成本必须另登记，不自动执行R17历史候选清单，也不读取旧保留test-dev30/作者test。

本轮0GPU/新增强CNN/SGD，不改变/占用R65训练或R66等待链。用project-python，资产只/data/Users/zrh/data/MobileIE-Neo经.cache；协议、目录/取样代码和计划、原始获取/技术错误/解码结果、过程/报告分别密集提交，阶段fetch/SSH推送/no-ff/指定作者保持。不改移动演示。此资料补外部有效性准备，不等于有新颖方法或完成论文目标。
