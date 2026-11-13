# OLiVES既有保留三片段的纯元数据资格

2026-11-13；Codex  (sqs_A100)。R9已保留作者test的video_43/71/83，继续禁止读取其图像；本阶段只核既有公开目录与摘要，不解除保留、不新增DEV或宣称独立最终完成。原R75/R76冻结源/预算/门槛保持。

固定原repo OLiVES-Lowlight/OLiVES、revision291ae19d061dedcff8e62f9b6d9d33d7ceeedbaf。先按已保存作者96 train/41 test清单和项目R3/R4/R4b/R9清单复算角色，不能把全部40个GT目录误认为40个作者test。现有元数据为31 train/9 test有双侧目录；作者test中已探索6个，剩余3个仅是待核保留ID。

仅8个串行匿名API GET：Frames/gt、Frames/input及三保留ID各两侧目录，limit1000、不recursive。每响应≤2MiB，无重试。保存Link/ETag/响应字节摘要及目录行数；若有下一页Link则标目录不完整并停止，不以首页面算完整基准。不调用resolve PNG、viewer、缩略图、下载器、npz/flow、模型或任何图像解码。公开HF工具没有固定revision逐目录/LFS清单能力，使用普通公开API；不读取登录令牌或其他用户资产。

只从目录LFS oid获得PNG内容SHA，与R3两个已读样例及R4/R4b/R9实际文件清单的已观察SHA全量比较。对每候选检查完整已列PNG元数据，而非只忽略重合首帧；任何side/hash重合均标不能作为严格未读视频。记录同名帧覆盖与原方案首16共同名的元数据，但不下载，且不能由零hash重合证明不同物理场景/拍摄take、预训练未见或未经未登记使用。R3保留example3也不读像素。

原始API文本缓存仅在zrh自有data/cache/datasets/OLiVES-reserved-metadata；仓库保存有限目录清单/角色/摘要/重合结果、异常/终态和过程。正式盲测资格仍须来源/语义独立与方法/评价冻结，不自动运行质量指标。DAVIS test-dev30 JPEG与Kodak保留亦保持未读。没有新实验家族或GPU，论文方法/自然动态/手机及独立最终目标仍未完成；分别小步提交和阶段推送。
