# MIE-0006｜vnrit 阅读与判断过程

Codex  (sqs_A100) <langxibielangle@qq.com>；2026-11-20。0上游应用执行/编译、模型运行/训练、GPU或Gate3实验。

## 身份与获取

固定[仓库提交](https://github.com/nlsidf/vnrit/commit/59bd6746c6cccde4197ccfabb1d60fb509127436) `59bd6746c6cccde4197ccfabb1d60fb509127436`；GitHub connector返回提交日期2026-08-04T01:06:27Z、消息`.`，本提交将Cargo.toml版本0.2.35改1.0.0。不是论文作者关联或MobileIE官方代码，账号nlsidf不等同真实作者身份。关联依据是main.rs:1275与1315显式称受MobileIE FST/通道注意力启发。

最初Gate1只有main.rs原件；Gate2通过GitHub connector核固定提交（首请求selector键错误被拒，改repo_full_name后成功），connector没有源码ZIP下载能力，故以公开codeload固定SHA补取完整9文件包。ZIP 114856B，HTTP200和摘要见[新回执](../../../../collection/receipts/mie0006-pinned-sourcezip.json)。main.rs在新包内的SHA与原Gate1原件42daff…f6b53完全相同，没有用默认分支覆盖旧快照。本项为已有身份补齐资料，不新增ID或改变Gate1计数。

## 实际阅读范围

- README全部420行，含架构字符图、增强/运行/依赖/参数/性能自述、限制提示与致谢。Cargo.toml全部，workflow全部；LICENSE核头部Apache2.0及尾部nlsidf2026署名，不提供法律解释或第三方许可完整审核。
- main.rs适用核心：502–566与609–700捕获/色彩/统计入口，733–1143缩放权重/融合MAD统计，1144–1898全部增强LUT/辅助/标量/SIMD/状态路径，1934–2030编码入口，2995–3013输出尺寸，3230–3553捕获—增强—编码—发送/缓冲/节奏。原186100B/4613行全件已归档，但未声称逐行阅读全部音频、信令、键鼠、安全或操作系统FFI。
- index.html仅video元素及1060–1125 WebRTC ontrack/连接/尺寸重置支持范围；不是全页JS/UI行为或浏览器实测。pako只核文件头第三方MIT AND Zlib标识，不审计整个压缩库。Cargo.lock额外缓存链接并按文本标记核378个package、0个git source；不是所有依赖实现审计。项目Python无tomllib的初始尝试在写入前停止，随后用标准库词法检查，无安装或共享环境修改。
- 无论文/PDF/作者图片视频样例/数据/模型权重或科学结果图；不虚构视觉审阅、全文论文或数值结果。README字符图已随原文实际阅读，不是图像展示。

## 任务与MobileIE的确切关系

主体是Linux/Termux X11桌面→BGRA/I420→手工增强→OpenH264→WebRTC→浏览器的远程桌面工具。Cargo描述与README均如此，输入是桌面显示，不是低照照片/视频数据集；触屏浏览器可控制不等于手机增强神经模型部署。

MobileIE只在两处注释出现。所谓FST-inspired实际为abs(当前Y−边缘修正blur)的分段整数Q8增益；不是原MobileIE多分支CNN、特征平方FST、HDPA学习参数、IWO训练或官方权重。attention-inspired实际是平均Y查LUMA_BOOST_TABLE，再乘全局锐化量；没有学得的池化—卷积—sigmoid注意力模块。全部9文件/依赖清单和核心路径没有神经网络推理/训练接线；不是以搜索无匹配代替正面核查。这是工程启发型关联资料，不能称“MobileIE Rust忠实移植”或独立被引论文。

## 核心信号与状态

捕获按当前帧转换到I420；缩放为预计算Q12 Mitchell–Netravali权重（源注释也称Catmull-Rom，名称不一致）加bilinear UV，tile/Rayon归约当前Y、TV与对prev的全局MAD。无运动向量、光流对齐、邻帧逐像素融合或未来帧索引。

prev在增强正常结束时写**最终增强Y**，运动提前退出时写**当前未增强Y**。所以下一帧MAD比较当前捕获Y与上一处理状态，不能一律称原始相邻帧运动；噪声/曝光变化/自身增强均可能影响全局MAD。这是静态机制推断，不是已测闪烁。运动因子查递推整数表v0=255、v[n+1]=floor(v[n]*209/256)，平均MAD整数>=14时因子<16而跳过blur/USM及后续色彩处理，仅存prev。不是估计连续几何运动或保证“消除时域闪烁”。

增强还含stack_blur、基于梯度的边缘回拉、分段非线性、纹理/全局亮度/色度/中间调乘子、clamp和局部tone curve。预先滤波不融合prev像素；历史仅通过全局MAD因子及色度统计缓存影响输出。RGB全三通道学习问题与YUV域手工锐化的任务/信息/监督不同，无训练数据优势或学习效果可核。

## 对注释主张的具体反核

独立[算术脚本](source_arithmetic.py)只转录字面式并算整数表/逻辑真值，没有导入或执行Rust/模型/媒体。结果见[算术记录](static-arithmetic-audit.json)，不是编译后SIMD仿真、视觉预实验或性能测试。

1. **标量与SIMD tone选择相反**：1814–1818的mask在grad>=16全1，XOR select取curve；1870–1872 scalar在grad>=16得到0，取enhanced，低梯度才取curve。按源注释的mask语义即可判断分支相反，不称同实现/逐像素等价。SIMD每行8像素组、末尾不足8标量；输出尺寸只取偶数，不保证宽度8对齐。实际像素/时序影响尚未测。
2. **非线性gain并非精确相同**：标量整数除法与SIMD近似斜率43/51/12不同。对0…255选中段的非负字面式计算有23个不同值，例如d=5为213/214、d=32为268/260；不把转录算术当机器整数所有路径验证，也不从系数不同推断感知严重性。
3. **名为denoise的blend在此定义下恒等**：w仅在|Y−blur|=0时为128，此时Y=blur；其他情况w=0。四舍五入blend两种情况都返回Y，SIMD判据相同。这里只说该blend，不说整个blur/增强对噪声绝无影响。
4. **亮度查表有阈值跳变**：avgY31→32时Q8值262→448；文案“单调暗处boost”不能替代实际函数。会不会在某视频引入可见波动尚未知。
5. **色彩不是始终保持或始终补偿**：CLI long_help称只动Y/chroma preserved，但1885–1895实际有UV饱和度乘子。tick=0算色度统计，随后每次加1模8；仅更新后tick=0且total_adj>0时对当帧UV boost，另外七次不做该boost。avg_adj为整数且可能0，不能说每第八帧必发生可见颜色跳动。
6. **首帧初始化不能按注释当已正确seed**：3309–3330先从仅有capacity的池Vec取缓冲、unsafe set_len，再在capture_to_i420成功填帧之前拷贝i420至prev。set_len不是写入像素；所示首轮路径没有当前画面已初始化的证据。apply_enhancement内兜底仅len不匹配才执行，而外层已设len。记录为静态未初始化读取/首状态风险，0执行、不尝试复现崩溃，也未作全项目内存安全审计。

## 默认、管线与性能口径

README说--height启用则默认enhance0.8；实际3240–3244是**needs_scaling**真才0.8，无resize且未显式enhance则0。指定同尺寸height不会因此自动增强。参数framerate默认24是请求/节奏值，不是增强吞吐测量。

README图称四阶段/容量4；实际capture+convert+enhance同blocking任务，encode另任务，send异步，两个视频通道容量均2。捕获满通道try_send丢当前帧，错误可重复上一帧；增强prev在是否成功送出之前已更新。因此预编码增强序列、实际发送序列和浏览器看到的时间间隔可能不同。send按固定frame_duration，不能以它反推实际帧采样或不闪烁。

README“硬件级编码”对应源码OpenH264API::from_source()/num_threads(1)软件库入口，不是硬件编码调用证据；只说编码器1线程，缩放仍可能Rayon并行。完整耗时还含捕获、格式/缩放、滤波、队列/丢帧、编码、传输、浏览器解码/显示。不将README的7.9MB二进制大小当模型参数或运行内存，也不对照MobileIE GPU/手机时延。此处没有可审查增强画质/时序/设备功耗基准。

原包含Cargo.lock与Apache2.0文件，较有版本/许可入口；README引用不存在的build.sh、patch段/git依赖，实际Cargo/lock不对应；workflow执行cargo update后publish，不是本固定lock的独立质量/性能测试报告。没有构建或依赖遍历验证，不根据CI定义称构建通过。

## 对二期的价值与边界

该资源对“已有人把MobileIE启发与运动闪烁同写在工程代码中”的重复线索有价值，但不是轻量视频MobileIE神经扩展的有效性证据。可迁移为手工USM/全局MAD门控的廉价工程对照、状态初始化/整数SIMD一致性检查和端到端采样/显示口径要求，不能把全局统计门控、设备语言转换本身包装成新算法。

后续可证伪控制：同一输入、固定codec/码率/采样与无增强、纯USM、单独MAD门控、逐帧MobileIE比较，区分关闭锐化与恢复时序质量；使用曝光变化/噪声/真实运动/切镜分组，检查原始prev与最终prev、阈值平滑、UV每帧/周期和宽度8对齐/标量尾差异。先明确修正上述条件路径再考虑运行；均为Gate3候选，不在本分支执行。不借此资源宣称论文创新成立或已有方案已消除闪烁。
