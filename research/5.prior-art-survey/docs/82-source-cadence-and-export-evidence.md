# 82｜来源帧率与导出证据的适用范围

2026-11-21；Codex  (sqs_A100)。本轮只看作者公开文档及已登记元数据，不打开视频/图片或重评模型。

[OLiVES v1 §3.1–3.4](https://arxiv.org/html/2610.20501v1)报告25fps、同步双机采集，RAW转RGB后作空间裁剪/短片切分；§3.3另述VOS子片，不能把VOS映射规则自动当作本项目LLVE Frames的抽帧记录。固定[HF README版本291ae19](https://huggingface.co/datasets/OLiVES-Lowlight/OLiVES/blob/291ae19d061dedcff8e62f9b6d9d33d7ceeedbaf/README.md)只明确输入/GT同文件名配对，没有逐帧时间戳/源RAW导出帧号映射。故25fps可作为**作者报告的名义采集率**，不是本项目独立测量的原始PTS。

[DAVIS2016正式原文§3](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Perazzi_A_Benchmark_Dataset_CVPR_2016_paper.pdf)明确将24fps限定在当时50个源。仅凭本项目采用DAVIS2017压缩包不能把全部新增源套用24fps。此次在[作者官方trainval归档](https://davischallenge.org/davis2017/code.html)仅读2016两个split文字成员，确认本项目SYN源中30个属于原2016、30个是新增；D1的bear/lucia在原清单，walking不在。因此walking的来源帧率在本轮仍未确立，不以第三方论文/重新编码视频的播放设置补证。

本地脚本/清单显示OLiVES取原文件00000..00015，DAVIS取连续JPEG前缀；源码linspace只选场景，不做时间抽帧，所有32个L1 held draw也连续。该层证据排除了本项目脚本的跳帧/重编号假设，但不证明作者导出和传感器时间间隔。具体来源、边界和未来协议时间契约见[完整核查](../../8.video-backbone/cadence-audit/README.md)。

已有三手机R49解码器位置显示Huawei/Pixel间隔有变化、iPhone近33.333ms常量；这是原解码器记录，不是独立sensor时钟、原视频包PTS重新核查或端侧推理延迟。新协议应分别声明帧预算、来源名义率/解码位置或实测时钟等级；没有根据将所有源统一视为实际30fps。

HF连接器读到当前README摘要，固定版本完整文本另由正常公开resolve URL取得715B并存自有cache；GitHub连接器只看作者仓库目录、未执行代码；匿名项目页不可达/猜测GitHub仓库404、原补充PDF重定向不可达均未绕过。CVF正式PDF§3正文可读，第三方搜索摘要只作定位线索，不作为帧率技术依据。旧nominal30fps JOD/原质量/D1指标与STOP原样保留，不因本轮新元数据回写既有实验。
