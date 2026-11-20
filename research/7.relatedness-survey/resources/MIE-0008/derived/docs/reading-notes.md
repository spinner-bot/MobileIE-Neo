# MIE-0008｜阅读范围、最终核对及证据边界

Codex  (sqs_A100) <langxibielangle@qq.com>；2026-11-20。0上游构建/二进制/模型/设备/训练/Gate3运行。前期状态按时间保留在[初读](initial-reading-notes.md)，完成取件和路径复核见[续读](acquisition-and-core-review.md)，不以旧pending描述代表当前取件状态。

## 已实际完成

固定e701008557c758876a3da75ed373d91dd959b1bf / Git tree ea53c880879249eb718eb0116cbfd7eb4eda87ad，元数据Nov12，作者账号KrakeCZ而非MobileIE论文作者。Git原件9494blob已逐路径/对象摘要核验，425个选中可读文本仅为准备范围，不能称9494/425份逐行阅读。全部原件byte扫描只有原Gate1四个MobileIE匹配文件；扫描也不是语义阅读。见[清单](archive-inventory.json)、[准备核验](preparation-validation.json)、[全件关键词扫描](full-archive-mobileie-scan.json)。

全文阅读：根README146行、根/Android两MIT文件、NightModePlugin接口/Manager、LiteIETflitePlugin、FastRetinexPlugin、AcademicYuvEngine、academic_yuv_jni.cpp两个函数、TimeLapseEncoder、TemporalDenoisingPlugin83行、LupaFrameAccumulator187行。Night Manager/Academic通过connector独立取得的原件还与Git包逐字节相同，见[交叉核对](primary-source-text-match.json)。

支持性适用阅读：SettingsManager默认/模型/缩放/分辨率/Lupa设置；ObjectDetectorStudioActivity的导入/选择/UI开关、CameraX/通道/预览/Academic与Lupa路由；PerceptionEngine注册/模型重载/生命周期/丢帧/Night/preview/OD支路；AgentCoordinator只核事件处理；LupaPluginManager适用主体与Pool尺寸/归还锚、LupaPlugin接口；Gradle构建/相关依赖/版本/CMake JNI编入；ThanksActivity确切致谢/BYOM条款。音频、LLM、FFmpeg整个实现、其他跟踪/眼动/SR插件、全部第三方/ABI安全和UI浏览器未全面审计。没有与MobileIE对应的已识别GT/作者增强输入输出媒体或科学图表，不拿应用图标/文字当视觉效果验证。

## 最终补核

- 普通Night默认关闭、默认选择非神经FastRetinex、无custom model；manual/autoContrast默认False，Academic也默认False/CPU。输入分辨率是实际文件shape，Academic按perceptionResolution（默认1024，设置范围160…4000）比例/对齐16另处理；不写成全项目固定256²。
- FastRetinex的已读实现只有固定ColorMatrix：brightness40、contrast1.5、saturation1.3，没有空间模糊/log/照明估计。名称和SSR注释不能证明运行了标准Retinex或MobileIE。
- 普通TFLite有单Bitmap输入/输出0的直接接口，无已识别的MobileIE架构/权重/转换脚本绑定。没有.tflite跟踪文件，两个音频ONNX和classes.jar均实际核为Git LFS指针，非可运行模型；没有据名称认定GPU/模型成功。
- 普通插件将非FLOAT32一律按byte处理，input/output dtype分别判但未见量化scale/zeroPoint处理。不能把weight FP16与输入tensor dtype混为一谈，也不能据此称所有量化输入均正确适配。具体BYOM类型未知。
- MobileIE文件名分支c/127.5−1与原官方RGB输入0…1不一致，但实际用户模型可额外包装/转换，故只能说**不能从名字确认忠实性，需对具体导出模型核对**，不称每个外来模型一定错。
- 输出自动范围启发式用首1000个float以内样本，不是全图/时间平滑；autoContrast开关False时不做min-max stretch，仍有阈值选单位。按帧范围可能改变映射是机制风险，未测可见闪烁。
- 热重载/异常存在原帧返回，500ms是release延迟常量/作者注释，不是测出的闪烁时长。manager过载raw分支存在，但当前主caller PerceptionEngine先以isProcessing丢帧，不能推断真实忙时一定进入manager raw分支。同路径释放后currentLoadedPath保留而interpreter空的条件路径与UI同路径选择/重载需在后续运行验证，不称普遍运行失败。
- 普通GPU创建在IO、run在manager专属线程；Academic Default协程也不固定线程，Mutex/Synchronized不等于EGL线程绑定。官方[GPU API](https://developers.google.com/edge/api/tflite/java/org/tensorflow/lite/gpu/GpuDelegate)给出线程/上下文约束；此为条件风险而非执行错误日志。QNN依赖实际声明存在，但反射构造、图委派、实际算子落点/后备和全NPU覆盖仍不等价。
- Academic JNI确实实现/编入，输入只复制Y三通道并最近邻resize，输出假定输入dtype/宽高/NHWC3，且手工offset的单位与Bitmap分支不同。不写成JNI缺失，也不将此处理与官方RGB实验等价。双Bitmap缓存只是生命周期/显示实现，不是神经时间记忆。
- TimeLapse会用合成PTS重排输出，异步提交复用Bitmap且没有在所读路径看到不可变快照/完成栅栏；实际渲染、吞吐和撕裂未知。输入Surface的CPU lockCanvas兼容条件见已核[MediaCodec官方参考](https://developer.android.com/reference/android/media/MediaCodec#createInputSurface())，当前源码/配置并不代替设备测试。
- 可选Lupa均值降噪是真实多帧代码：cache=True时按minFrames/250ms等可变条件收集历史batch并清空，CONFLATED可丢帧；不保证每次恰5帧。Temporal插件只标category/minFrames，requiresMultipleFrames沿用接口False；cache=False时最终process以单帧均值退化。默认lupaVisionDebugMode=False，只有开启且latestResult存在才回送Night/检测支路；不是默认MobileIE时间模型。SkyMode帧差在OD预处理，不能冒当Night输出稳定器。

## 材料可支持与未知

本项充分支持“社区在相机工程中显式标注MobileIE并提供BYOM接口/后备/录像/可选均值路径”，不支持已取得忠实MobileIE TFLite、模型质量复现、训练数据/先验一致、全帧时序改善或某手机/NPU实测。参数/算量/模型内存随外来模型未知；tf.run范围的计时源代码不是完整pipeline的数值日志，Gradle注释INT8目标8–25ms不是结果。

两份MIT文件实际存在（ggml authors与AiChat contributors），根README也有MIT/GPL占位及第三方声明；Thanks明确BYOM权重许可另核。只记录证据/范围，不作法律解读或全面软件依赖许可审计。没有模型全文/权重的获取失败问题，因为这里实际是导入用户文件的通用接口，未识别出可公开下载的特定MobileIE转换件；不伪称“权重齐备”。

科研可迁移价值在于输入/后处理/调度/编码也可能造成表观不稳定，及低成本均值控制的明确存在。重复风险不是一个已证明的MobileIE学得视频算法，而是通用逐帧封装、GPU/NPU选择、双缓存、输出映射和简单平均都不能单独构成新颖性。具体可证伪控制在本项报告列出，只交后续实验分支。
