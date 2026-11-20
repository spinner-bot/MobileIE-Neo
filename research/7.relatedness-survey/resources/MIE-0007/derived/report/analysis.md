# MIE-0007｜ndk-vnrit：Android编码变体，非神经增强移植

Codex  (sqs_A100) <langxibielangle@qq.com>；2026-11-20。完整固定项目适用核心静态分析，0上游编译/二进制/模型/设备运行或Gate3实验。

**该项目把编码器接到Android NDK MediaCodec，增强仍是MobileIE启发的CPU整数USM/全局统计门控。** 它提供Android媒体管线参考，不能证明MobileIE网络已经视频化、增强在NPU运行或时序问题已解决；与MIE-0006属于同作者同族工程线索，不计两次独立算法验证。

固定[nlsidf/ndk-vnrit@b169f207](https://github.com/nlsidf/ndk-vnrit/tree/b169f207b3766a4ab037bfd5202860ea9f553d43)，Cargo0.1.28，提交元数据2026-07-25。该快照早于MIE-0006 Aug4，不推断Git祖先。13文件完整包与原Gate1 main.rs单文件SHA一致；[身份](../../identity.json)、[来源](../../sources/manifest.json)、[过程](../docs/reading-notes.md)、[差异/静态核查](../docs/static-port-audit.json)、[结论证据](claims.json)均保存。

## 机制与继承边界

增强块在明确去注释/空行和统一SIMD方法名后与前项文本相同，首帧prev初始化片段亦相同；不宣称不同API编译语义已验证。沿用的全局MAD不会区分运动/噪声/曝光或自身增强，无历史像素融合/对齐、没有未来帧索引。源码中的tone curve标量/SIMD相反选择、局部denoise恒等、gain与luma阈值、每八次色度条件及捕获前prev初始化风险仍存在于本快照；影响未实测。

MediaCodec路径增加动态库/符号加载、video/avc编码构造、I420→NV12与拷贝输入、CSD/SPS/PPS输出拼接及参数调整。它是编码硬件接口，不是神经模型后端。按MIME创建首选编码器不记录具体组件，缺实际硬件选择/能力日志；官方[MediaCodec](https://developer.android.com/reference/android/media/MediaCodec#createEncoderByType(java.lang.String))与[MediaCodecInfo](https://developer.android.com/reference/android/media/MediaCodecInfo#isHardwareAccelerated())语义也不支持仅凭API名字认定某个设备实现。当前API参考不冒称截止快照的历史设备测试。

## 不能混用的证据

|资料|能支持什么|不能据此声称|
|---|---|---|
|MediaCodec源代码|有实际Android NDK编码接线、NV12缓冲与状态管理|增强上NPU、必选硬编、完整端侧实时成功|
|README资源表|作者自述Snapdragon835/720p500kbps、RSS50–80MB、视频2–3 CPU核约1.5GHz、目标24fps|当前NDK版本/增强开关下独立测量或与MobileIE公平ms比较|
|test_mc.rs及ELF|存在dummy NV12编码探针，二进制头是AArch64 ELF|探针已在某设备成功、源/二进制匹配、增强画质/时序已验证|
|models_new.json|空0B文件|模型权重、神经架构或实际模型发布|
|文档/许可|README仍称OpenH264/MIT，Cargo称Apache-2.0；包无LICENSE|文档已与后端同步、授权条款已经核清|

输入编码PTS用墙钟微秒，输出收集却未把每buffer的PTS用于传输分组；发送仍固定duration。drain可收集多个可得buffer，不等于已证明某次包含多幅图像，亦不能证明capture到browser保留了真实每帧时刻。掉/重帧、色彩转换、codec/profile/码率和显示节奏都会改变结果，增强前后的公平性需在完整pipeline上核。

## 对二期研究的价值

应借鉴的是部署成本与证据的分层：CPU增强、NV12转换/拷贝、实际选中codec、编码等待/缓存、传输与显示分别记录。同族项目的传统锐化可列廉价工程控制，不将编程语言、MediaCodec或全局MAD门控当新学习算法。原始MobileIE参数/FPS、README应用RSS、硬件编码指标没有共同口径，不能拼成效率胜出。

后续可证伪控制为同输入/状态/增强开关、同码率/profile/分辨率与真实时间轴下比较OpenH264/MediaCodec，并同时记录预编码图像与浏览器输出的时序/空间质量、codec名/能力、掉重帧、warmup和总RSS/能耗。先解决共用初始化/整数分支问题，加入无增强/纯USM/门控及逐帧MobileIE。所有设备行为、精度/时序和总成本仍unknown；这里只提出后续实验分支候选。
