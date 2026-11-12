# R52｜DRV原始资料的元数据阶段

登记：2026-11-12T19:28:50.736141+08:00；作者：Codex  (sqs_A100)。补实际低光外部验证获取，0GPU/0CNN；R47最终seed0仍运行，不能用其部分结果选资料。

已固定官方源码e3e81e742489f997bba30bf331c8d3fa58ddc2b0与train129/val24/test49，互斥合计202静态场景。当前只读作者论文和ZIP目录，不读任何RAW/PNG数据成员；独立最终test49与dynamic23不打开。预处理VBM4D_rawRGB是16bit三通道/65535、已含既定时序滤波，不可当普通单帧输入或未经处理的原始视频。

仅对官方GCS short1–5.zip及long.zip做HEAD和central-directory/ZIP64元数据，最多4MiB/归档，验证206/Content-Range/If-Match ETag；复用既有RemoteZip，不下载整个百GB归档、不绕过认证。完整索引存自有data，元数据清单/摘要/统计入Git。作者公开论文PDF存自有data，仅解析实际获取后再阅读§3–4采集/预处理/限制，不将论文取得等于视频已验证。

若原始RAW与官方split可完整对应，只按train排序索引[0,n//3,2n//3,n−1]保留四个未来探索候选；目录不含内容选择。获取RAW像素、render/黑电平/曝光/色彩/时间规则需在后续独立登记后执行，现阶段没有授权该脚本自动下载像素。static长曝光GT不能给dynamic Mxxxx作PSNR；静态夜间相机子集也不是完整标准动态benchmark或手机实测。

并行已只读确认Night Wenzhou论文v5 §III-D/TableIV为夜间GoPro/DJI未配对视频，但作者当前README链接匿名404；唯一issue无替代数据链接，暂不重复请求/使用未证实镜像。此获取失败不阻止DRV元数据。所有资产只进zrh项目data，源码/元数据/过程/报告分别小步提交并阶段推送。论文创新/实际外部验证尚未完成。
