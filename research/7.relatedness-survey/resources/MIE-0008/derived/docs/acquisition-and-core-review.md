# MIE-0008｜归档完成与核心路径续读

Codex  (sqs_A100) <langxibielangle@qq.com>；2026-11-20。接续initial-reading-notes的历史状态；当前仍未Gate2结项，不覆盖初读记录。0上游构建/运行/模型/设备实验。

## 原始资料已取得的精确范围

两次codeload尝试分别128MiB大小截断、768MiB上限重试中的网络read timeout；回执与第一份prefix均保留。第二份异常时没有完整body留存，不能称完整文件已下载。改用zrh自有cache/sources/MIE-0008.git，以bare Git fetch --depth=1精确revision，再git archive生成84,570,834B源码ZIP。没有checkout、LFS smudge、构建或执行上游代码。

9494个Git跟踪blob/385,357,828原始字节全部对照固定API tree：路径集合相同，每个blob按`SHA1('blob '+长度+'\0'+原件)`与API摘要匹配。实际commit^{tree}=ea53c880879249eb718eb0116cbfd7eb4eda87ad；API响应sha字段回显请求commit e701…，因此单独保留真正Git tree，不直接把API字段当tree。完整核验结果见collection/receipts/mie0008-git-pinned-source.json。

资料准备实际清点9494件，按列明prefix只链接425份Android应用/根说明/构建/许可文本，合计6,349,966B；不缓存全部Whisper/Vulkan第三方源码副本，不称425份均全文读过。四份原Gate1 codehit与新归档对应文本摘要完全相同。Git跟踪LFS指针属于原件，未另获取其大模型内容；不据文件名扩展称模型已运行可用。HTTP后缀启发式把libs.versions.toml归档为.json，实际内容仍TOML并按源格式阅读，不把文件名当有效JSON证明。

## JNI/依赖核实

起先media_engine_jni.cpp没找到Academic方法，仅说明该文件不是声明位置，不能据此称JNI缺失。实际实现是academic_yuv_jni.cpp，两个完整JNI函数已全文阅读，CMakeLists确实将其编入media_engine。额外独立原件也按固定SHA取得。

输入仅读取CameraX的Y plane，用整数最近邻下采样，逐像素把相同Y值写R/G/B三个输入通道；float为Y/scale−offset，byte直接Y。没有在此路径保留U/V或做标准RGB色彩还原，亦未见Y范围适配。不能把此输入等价于官方MobileIE RGB视频测试；未知BYOM可能自有适配，但需模型/元数据证据。

输出JNI按输入isFloatModel标志、输入宽高和假定NHWC3布局读取模型输出；float计算`(v+offset)*scale`、旋转写ARGB；没有独立检查输出dtype/尺寸/量化元数据。它和普通Bitmap插件的`v*scale+offset`手工输出参数语义亦不同（各自独立settings）。实际越界/画质影响未运行，不作全JNI安全审计。

Gradle+版本目录实际有LiteRT1.0.1、qnn-runtime/qnn-litert-delegate2.47.0声明；Academic反射候选不是“根本没依赖”。注释中的Snapdragon8Gen3、INT8目标8–25ms仅目标/作者说明，未取得设备运行或完整fallback/算子落点日志。反射HTP setter内部失败会忽略，delegate构造成功与Interpreter图委派成功、实际全NPU不是同一证明。GPU/CPU/NPU是选择条件，默认Academic CPU。

当前官方GPU API要求适当EGL/创建调用线程一致性，参考保存official-gpu-reference.json。普通插件在Dispatchers.IO创建/重载、manager专属线程run，代码仅Mutex串行，没有在所读路径显式共享EGL；Academic调用者是Dispatchers.Default，主线程切换/接收后不能由Synchronized保证线程固定。记录为条件兼容性风险，未称已发生特定设备错误；默认CPU不受该GPU路径假设影响。

## 实际相机/显示调用

PerceptionEngine:281–309注册两个Night插件并响应reload；545–590有生命周期/热管理/忙时丢输入，普通night打开且非academic时才activate/enhance；处理后581…625送preview/检测。默认night=False、retinex、无自定义权重；实际默认Retinex也只有在启用night后才进入。

ObjectDetectorStudioActivity普通模式RGBA，academic模式YUV/不建原Preview；2420–2441设KEEP_ONLY_LATEST，2511–2533 academic将ImageProxy放CONFLATED通道，最终在Academic finally close；2660–2707以Dispatchers.Default消费、可再缩放、传PerceptionEngine并将复用ping-pong Bitmap赋UI。不能把双缓冲/通道称神经时间建模，也不能以注释“立即释放camera线程”证明相机硬件吞吐/内存零复制。

TimeLapseEncoder完整源码已读，创建MediaCodec surface，CPU Canvas画输入，输出写MP4时**将PTS重写为固定frameDuration的合成时间**，并不保留真实相机采样间隔。Academic用IO协程异步提交复用的双Bitmap，没有看到逐帧不可变快照/完成栅栏；需验证生命周期而非从ping-pong注释推断不撕裂。SDK的MediaCodec输入Surface对CPU lockCanvas还有兼容性条件，须与当前代码一起留未知，不称已经录制成功/时序流畅。

## 另有可选多帧功能，不能漏报或混用

项目并非完全没有时间处理。实际读TemporalDenoisingPlugin全文83行，LupaPluginManager适用主体及Settings/调用锚：有可选lupa_temporal_denoise，minFramesRequired5、expectedFpsHint15；实际算法逐像素简单RGB均值，无运动对齐，注释静态图100%保留/sqrtN有假设。fpsHint不是实测。Lupa cache配置可变且默认manager cache=True，关闭cache时多帧插件被以单帧list调用；不能宣称所有使用都恰有5帧。

普通相机在lupaVisionDebugMode=True（默认False）且有latestResult时才把Lupa输出送PerceptionEngine，之后可选Night插件；其异步latestResult可能与当前frame不同时刻。另见SkyMode下FrameDifferencePreProcessor的调用属于对象检测输入预处理，未把它当Night模型自身时间稳定化。需要将这些支路与“单张TFLite低照推理”区别，不能宽泛说整个app没有多帧功能或已经是MobileIE学得视频方法。

继续：核适用Lupa积帧/时间条件、模型导入/元数据与作者性能/授权证据，完成本项主要结论账本及第二遍Gate2。当前文档为过程资料，不提前发布整体调研结论。
