# MIE-0008｜Android相机BYOM封装：关联明确，模型与时序效果未证实

Codex  (sqs_A100) <langxibielangle@qq.com>；2026-11-20。适用核心静态项目分析；0编译/模型/设备/训练/Gate3实验。

**该社区项目确实把MobileIE列入低照相机插件，但当前证据是通用TFLite/BYOM和媒体工程，并非可核验的MobileIE视频模型或性能复现。** 它的价值是揭示输入信息、输出映射、线程/调度和录像时间轴的混淆因素，并提供可选多帧均值控制。

固定[KrakeCZ/AIchat-Android-Local-AI-Agent-@e7010085](https://github.com/KrakeCZ/AIchat-Android-Local-AI-Agent-/tree/e701008557c758876a3da75ed373d91dd959b1bf)，提交元数据2026-11-12。它是基于llama.cpp的独立实验Android应用，非MobileIE作者项目。[身份](../../identity.json)、[原件与失败回执](../../sources/manifest.json)、[阅读过程](../docs/reading-notes.md)、[逐结论证据](claims.json)明确实际范围。

## 四层证据必须分开

|层级|固定资料支持|没有证明|
|---|---|---|
|关联|全9494blob扫描中4个源码文件共9处MobileIE字符串，插件/设置/UI/致谢相符|学术被引论文、官方网络移植或作者验证|
|模型|用户导入任意TFLite，读取tensor0/输出0、可试GPU和Academic QNN路径|确切MobileIE架构/权重/转换件、量化/预处理忠实性、完整委派覆盖|
|视频工程|CameraX、异步管理、双缓存、TimeLapse和可选多帧均值|模型学得的时间恢复、连续画质提高、真实采样时间保持|
|成本|有局部tf.run计时代码、后端配置和8–25ms目标注释|某手机上可复核的质量/时序/总延迟/峰值RSS/功耗结果|

完整Git包9494文件与每个API blob摘要一致，425文本是选中准备范围，不是425份全文精读。LFS指针内容不等于取得大模型；未识别特定MobileIE TFLite权重/科学GT或增强演示。两次HTTP归档失败保留，后续bare Git归档已完成；取件成功不当运行成功。

## 主要机制与反证

普通night默认关闭，开启后默认FastRetinex是固定颜色仿射/饱和矩阵；选择TFLite后仍需custom文件。插件根据文件名mobileie选择[-1,1]输入，外来模型的真实预处理/布局/量化须独立核；不能因UI名称直接认定等价于官方MobileIE RGB。输出会按单帧首1000float样本以内的范围选映射，只有用户另开autoContrast才做min-max stretch（默认关闭）；该处理不含时间平滑。

热重载期间可返回原帧，异常后备也是原帧；同路径release后未清路径的条件初始化缺口保留。manager忙时原帧分支存在，但主PerceptionEngine本身会先丢忙帧，不能从分支存在推断实际每次交替。GPU初始化/调用线程缺显式一致性保证，按[官方GPU线程/上下文语义](https://developers.google.com/edge/api/tflite/java/org/tensorflow/lite/gpu/GpuDelegate)是需验证的兼容条件，尚无实际错误日志；CPU/GPU/QNN选项不是实测后端证明。

Academic是另一输入协议：JNI只取Y、最近邻resize并复制为3通道，没有原始U/V；输出假定输入dtype/尺寸/NHWC3。它实际存在且CMake编入，不能误称缺JNI，也不能视为官方RGB同条件结果。默认1024目标分辨率还可被外来tensor形状/对齐改变，不是统一固定256²。双Bitmap与线程锁没有增加神经时间信息。

TimeLapse会重写合成PTS，CPU surface绘制/异步复用Bitmap还需验证兼容与生命周期。相机KEEP_ONLY_LATEST、CONFLATED、热管理和忙时丢帧改变显示采样；因此录像“流畅”不能代替按真实时间对齐的闪烁指标。

项目另有可选TemporalDenoisingPlugin，实际是历史帧逐像素RGB平均，无运动对齐。名义minFrames5/fpsHint15分别是配置/提示，cache/250ms超时/丢帧可改变实际N；关cache会退成单帧，默认不回送检测支路。它可以是传统工程对照，不是默认MobileIE学得时序方法，也不能声称整个应用完全没有多帧功能。

## 可迁移价值与后续反证

输入/后处理与调度可能改变表观稳定性，不能把所有变化归为MobileIE逐帧网络。可迁移的是BYOM加载边界、相机/录像口径核验和均值对照；不可直接迁移的是模型身份、无完整日志的目标速度或未经检验的画质收益。简单逐帧封装、delegate选择、双缓存、均值/自动对比度本身不足以建立二期算法新颖性。

后续实验分支可检验：

1. 固定官方权重及可追踪导出哈希，逐项验证RGB/灰度信息、归一化、布局、量化、clamp和输出映射；禁用文件名/范围猜测，先证明一帧函数一致性。
2. 固定RGB视频、真实时间戳/状态和相机曝光条件，比较逐帧MobileIE、固定封装、强时间后处理/均值控制；分运动/曝光/噪声/切镜，区分少改变输入、过度平滑与真实细节/时间收益。
3. 记录所选CPU/GPU/QNN与实际委派/回退，核创建/调用线程及同路径重载；测采集—预处理—推理—后处理—显示/编码的首帧/稳态、峰值内存/能耗/热与掉重帧。预编码图像和合成PTS录像分开评价。

当前模型权重/训练先验、算子覆盖、运行兼容、连续时序/空间质量、总设备成本仍unknown。两份MIT文件存在，README有许可占位和第三方声明；外来权重授权另核，未作全面法律/依赖审核。上述为可审查的静态分析与候选问题，Gate3尚未执行。
