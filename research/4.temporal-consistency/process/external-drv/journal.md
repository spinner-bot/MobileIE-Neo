# R52 DRV元数据阶段

记录：2026-11-12；作者：Codex  (sqs_A100)。源代码/协议/来源读记/原始过程/结果/报告分别提交，数据本体只存zrh项目data。

19:28登记元数据协议，固定官方源码e3e81e742489f997bba30bf331c8d3fa58ddc2b0及互斥129/24/49静态split。train固定排序0001/0072/0138/0202为未来候选，仅元数据选择；未读RAW或PNG。

19:32获取作者10页PDF（20,897,057字节、SHA见paper-acquisition.json），实际定向读§3–4的采集、VBM4D/ISP与训练定义，和官方train.py/test_video.py交叉核查。原始RAW、官方VBM4D时序预处理输入、sRGB输出并不等价。读记note42保留限制，未声称复现作者实验。

archive-metadata-attempt001解析失败：对帧文件名与场景目录混淆，0完整归档；failure/runtime/log原样保存。修订优先目录，attempt002正常完成六个官方归档目录，33.598秒，总传输central-directory元数据1,823,497字节，HEAD/Range均为正常公开接口，没有下载整包。

复查发现attempt002剩余语义问题：动态Mxxxx目录下0001_*.ARW仍被fallback误标静态0001。原始成员名/CRC/offset/大小有效，但当时matched_static_scene标签已被替代。用classify_drv_catalog_r52.py对现有索引本地重分类，无新增网络、无像素读取；严格按目录分类。classification001/catalog.json为最终权威清单：long改44标签、short5改114，short1–4不变；202静态scene在raw+long均齐，实际动态目录22个，不与源码枚举23个混同。

完整索引位于.cache/datasets/DRV/archive-index/classification001，原始attempt002索引保留。四个候选短RAW数量111/109/110/64，只是目录数量；没有获取这些数据成员。独立test49、动态像素均未打开，0CNN/0GPU/0图像或视频解码。后续读取/渲染像素必须另定曝光、黑电平、色彩、量化与时间协议，本阶段不自动续到像素实验。

2026-11-12T22:04:00+08:00补核：本日志此前提及的note42未实际存在于当前Git；重新读取已保存作者PDF文本§3–4/4.1，补建research/5.prior-art-survey/docs/42-drv-raw-isp-and-temporal-preprocessing.md。保留真实补记时间，不把旧引用当当时已经落盘的证据。原始RAW与VBM4D时域预处理再逐帧网络的区别仍成立；本次0新成员获取/解码/CNN/GPU，不改变元数据状态或旧失败记录。作者：Codex  (sqs_A100)。
