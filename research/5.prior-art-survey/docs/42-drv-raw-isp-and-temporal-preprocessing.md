# DRV：RAW、ISP与时域预处理边界

补记：2026-11-12T22:04:00+08:00；作者：Codex  (sqs_A100)；原因：外部DRV过程日志19:32引用note42，但当前Git中没有对应文件；本次按实际保存的作者全文重新核对方法段并补齐，不回写成此前已存在的读记。

实际读[作者全文](https://vladlen.info/papers/DRV.pdf)§3–4及4.1；自有副本与获取记录见[paper-acquisition.json](../../4.temporal-consistency/process/external-drv/paper-acquisition.json)，10页PDF及逐页文本都在zrh项目data。没有重新下载论文、读取RAW/PNG数据成员或复现网络。

作者用Sony RX100 VI采集RAW序列，约16–18FPS；静态序列配长曝光参考，动态序列没有可用于量化的清晰GT。202静态序列的划分应与具体实现的scene清单一起核对，不由论文百分比重造split。

方法将Bayer拆到RGB、合并两绿像素、减黑电平并binning，按曝光差缩放/clip，再用VBM4D减时间噪声；其后ResUnet才逐帧输出sRGB。静态两帧共享GT的孪生感知/一致性训练也有明确先例。单帧网络不等于无历史流水线，预处理的时间信息必须计算在内；没有复现该预处理或作者性能。

## 本地资料与下一步约束

当前只有六个ZIP中央目录，权威重分类是[classification001/catalog.json](../../4.temporal-consistency/process/external-drv/classification001/catalog.json)。静态129/24/49 split互斥；四个train候选0001/0072/0138/0202是在像素读取前按元数据确定，不能按图改选。原attempt001解析失败、attempt002动态文件名误标及后续本地重分类均保留；不能用旧matched_static_scene代替权威目录标签。尚无RAW/PNG成员下载或解码，不能称外部量化验证已完成。

未来若获取少量RAW配对，须先登记固定成员/时间索引、曝光增益、黑白电平、相机白平衡、binning/ISP、颜色编码、位深与量化规则。原短曝光、官方VBM4D_rawRGB和相机sRGB属于不同输入链，不能混作同一个单帧基线；不用GT均值校正输出、不给未经确认的动态长曝光当清晰GT。不把标称传感器分辨率/缓冲帧率当本项目手机输入或实测速度。

这是已有资料的来源边界与过程缺项修补，0新CNN/SGD/GPU、0新数据像素。Night作者部分备份另按原下载→目录链推进，不能从尚未完整的RAR推断输入/增强输出身份或宣称有自然夜间GT。论文创新、真实夜间有效性和端侧实测仍待建立。
