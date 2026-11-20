# MIE-0008｜进行中的初读记录（不是结论或Gate2审查）

Codex  (sqs_A100) <langxibielangle@qq.com>；2026-11-20。当前唯一活跃资源，未结项，尚未开始下一项。以下为已实际阅读的局部固定源码，须在JNI/相机/Gradle及完整包核验后形成最终判断。0上游构建/运行、模型/训练/设备/Gate3实验。

## 获取状态

固定e701008557c758876a3da75ed373d91dd959b1bf，GitHub元数据2026-11-12T01:10:54Z。初始codehit四文件保持；第一轮ZIP到128MiB被本地大小上限截断，回执size_limit_incomplete和原始prefix保留，不能作为完整ZIP。768MiB上限重取仍在进行。公开递归tree已实际取得HTTP200/3306713B，truncated=false，10647entries/9494blobs、总blob385357828B；包含大量Whisper/Vulkan构建产物。既有资料准备脚本新增可选text-prefix，全件仍核哈希/清单，文本副本聚焦Android应用；当前尚未成功准备完整ZIP，不能声称源码包已收全。

已通过GitHub connector另取得固定AcademicYuvEngine与NightModePluginManager全文，分别留原始primary JSON。初始四文件/元数据与失败回执已分别提交；逐帧入口先读LiteIETflitePlugin全文，SettingsManager相关night设置、ObjectDetectorStudioActivity设置/分析器锚、ThanksActivity确切MobileIE致谢。

## 已读路径中的条件（尚待全链复核）

- 默认isNightModeEnabled=false、插件night_retinex、custom model path=null、manualScaling=false、autoContrast=false；academicNightVisionMode=false、academic hardware默认CPU。不能将UI列有MobileIE说成默认已运行MobileIE模型。
- LiteIETflitePlugin以用户文件mmap构建TFLite；只读input/output0，识别简单NCHW/NHWC，FLOAT32以外一律byte，实际量化参数/模型内容尚未知。文件名含mobileie且非manual时使用c/127.5−1，其他使用c/255；和官方RGB模型预处理的关系必须对照具体转换模型，不能凭MobileNet注释认为适配已正确。
- 插件逐Bitmap resize、填buffer、tf.run、重建/回缩；计时只包tf.run。输出前1000float样本判断范围；**只有autoContrast开关打开**才做动态min-max stretch（默认False），但默认仍有数值范围启发式选择scale/offset。没有在当前wrapper看到帧间稳定化状态，不能据此推断未知模型的内部variable状态。
- process热重载在Dispatchers.IO启动init，isInitializing期间直接返回原始input，代码也自述可能闪回暗画面。NightModePluginManager用专属single-thread executor进行process，过载/异常也返回原input；Mutex只串行化访问。GPU创建/调用线程与EGL约束需官方语义核验和实际调用路径审查，尚不宣称某设备实际报错。
- manager同路径reloadActivePlugin只release而不init；plugin releaseInternal清空interpreter/delegate但不清currentLoadedPath，process仅path不同才触发init。同路径释放后可能一直原帧的条件路径已见，待调用侧核实；不等同已观察到运行故障。
- AcademicYuvEngine是另一入口：NHWC3通道假设、按目标比例对齐16、可resizeInput、GPU/反射QNN NPU候选及后备；真实依赖和硬件是否可用未核。wrapper有JNI Y平面→输入/输出→ARGB、双Bitmap ping-pong及异步TimeLapse。双缓存/录像并非时序恢复算法，JNI色彩/数据型别与可变Bitmap生命周期仍待阅读。
- ThanksActivity明确BYOM/独立解析与算法实现的致谢，并提醒论文不等于权重许可；不能从致谢自动认证MobileIE权重身份。tree内没有.tflite，两个音频.onnx为132/133B入口，尚未按完整包查看其内容；不拿扩展名当已取得可运行模型。

下一步：完成归档/清单；读night管理器调用、Academic JNI、TimeLapse、Gradle依赖/模型来源及适用许可；核相机真实输入/丢帧/显示链，固定本项证据与局限后再作第二遍Gate2。以上均为过程记录，不提前给结项状态或总体新颖性结论。
