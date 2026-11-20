# MIE-0007｜ndk-vnrit 阅读过程

Codex  (sqs_A100) <langxibielangle@qq.com>；2026-11-20。0上游编译/二进制执行、MediaCodec调用、模型/训练/GPU/Gate3实验。

## 身份、资料与实际阅读

固定[项目提交](https://github.com/nlsidf/ndk-vnrit/commit/b169f207b3766a4ab037bfd5202860ea9f553d43) `b169f207b3766a4ab037bfd5202860ea9f553d43`，GitHub元数据2026-07-25T10:52:10Z、消息mem-fix，实际该commit diff仅Cargo包版本0.1.27→0.1.28；不能据提交标题推断修好所有内存问题。此快照比MIE-0006 Aug4早；“同作者变体”不等于已证明按时间接续或Git祖先。

Gate1单文件之外，以公开codeload固定SHA补全349584B/13文件包，回执、时间与摘要入库；main.rs与原单文件SHA80d425…57cff一致。GitHub connector用于提交元数据，ZIP用其不提供的公开归档下载。新增仅当前既有身份的资料，不增静态ID。身份为nlsidf独立项目，不能当MobileIE作者官方项目。

README英文全篇已读；Cargo.toml全篇与相较MIE-0006的44行diff、HTML37行键盘标签diff、main768行完整diff全部阅读。新增MediaCodec主体main.rs:1914–2325、调用任务3702–3809及相关输入/输出/时间戳上下文实际静态核查。main共4887行，不声称逐行审计全部网络/音频/FFI安全。test_mc.rs测试入口全文读过；它生成一张640×480全128的NV12 dummy输入并打印结果，不是图像增强/连续视频质量或时延基准。

两份附带API笔记按标题、目录及适用sample/codec部分检查：api.md针对webrtc0.20.0-rc.1，Cargo为rc.4；openh264-api.md针对OpenH2640.9.3，但当前Cargo/代码已移除其编码调用。不能拿笔记当当前依赖完整实现或运行证明。vnrit.md相较前项中文README只在MIT/Apache许可与6.6/7.9MB自述有三处差别，主体阅读复用且保留diff。未逐行重读压缩JS、全部API示例和HTML样式/交互，也未浏览器运行。

完整包有Cargo.lock（额外缓存链接、词法核365个package）、570008B test_mc ELF文件、0B models_new.json，没有LICENSE、科学图表/论文/模型权重/输入输出媒体或测试日志。ELF仅核magic/class/endianness/e_machine183与哈希，未执行/反汇编或证明其来自所附源；可执行文件含成功打印字符串不是已经测试成功的日志。空models_new.json不是发布神经模型。没有可实际观看的科学图像/视频附件，不伪造视觉审阅。

## 与MobileIE、MIE-0006的关系

MobileIE仍只出现在1266/1306注释，对应手工非线性gain和全局亮度LUT的启发，不是运行MobileIE CNN/FST平方/HDPA/IWO权重。整体仍是X11屏幕远程传输，不是自然低照数据集的视频增强训练。

对从build_motion_table到编码器前的增强块，去空行/注释、trim，并仅将.mul_widen方法名统一成.widening_mul后，与MIE-0006逐行文本完全一致；捕获前prev初始化片段也一致。原样与归一化比较均留[静态审计](static-port-audit.json)，不是证明两个不同依赖API在编译后语义相同。未把两个非同SHA的整仓/模型称字节相同。

因此MIE-0006关于整数表达式的判断适用于此选中源码：全局MAD状态不是光流/像素历史融合；正常prev写增强Y，提前退出写原Y；MAD14起关闭增强；denoise blend局部恒等、luma31/32跳262→448、23个scalar/SIMD gain表达式差别、tone选择相反、UV仅特定每8次条件boost、首帧捕获前set_len后拷贝prev。重新核本项关键位置：1591–1597、1807–1816、1862–1866、1876–1886、3587–3620。具体原式与算术在[MIE-0006审计](../../../MIE-0006/derived/docs/static-arithmetic-audit.json)，复用而不虚构第二次数值实验。性能/感知影响及初始化运行后果仍未验证。

## MediaCodec实际增加什么

Cargo取消OpenH264，增加NV12格式类型，代码通过dlopen libmediandk.so及dlsym取得NDK MediaCodec函数；1958–2005尝试system/apex/vendor路径，缺库或符号报错，当前路径没有OpenH264后备。2084–2138按video/avc创建编码器，设宽高/码率/framerate/10秒I帧间隔、color21/NV12、stride/sliceheight及latency1请求，configure/start后在编码blocking线程内保留。latency1是配置请求，不是1ms或1帧实测保证。

CPU增强仍先完成，再经libyuv I420→NV12写预分配nv12，copy_nonoverlapping拷到codec输入buffer。没有NNAPI/神经NPU路径，也不是整个pipeline零拷贝。MediaCodec编码硬件与神经增强硬件是不同证据对象。

其构造只createEncoderByType，没有记录选中codec名、硬件属性或具体SoC/驱动。当前官方[MediaCodec参考](https://developer.android.com/reference/android/media/MediaCodec#createEncoderByType(java.lang.String))说明按类型创建首选编码器；[MediaCodecInfo参考](https://developer.android.com/reference/android/media/MediaCodecInfo#isHardwareAccelerated())另提供硬件/软件属性。由此不能从一次API调用认定选中硬编，更不能认定增强跑NPU；当前依赖文档只是语义核验，不是July25设备运行事实或新增调研方法身份。实际检索与关键摘录留[官方参考回执](official-media-reference.json)。

## 时间戳/缓冲与控制条件

2214–2235输入PTS取encoder epoch墙钟微秒，在调用端queue input；dequeue input可等待10000µs，没有buffer时返回Ok。输出drain以5000µs一次、200ms循环截止收集可得buffer，处理format changed取csd-0/1与SPS/PPS，并做offset+size边界检查，非每帧全部固定200ms。set_bitrate和force_keyframe请求setParameters，未检返回值；实际设备接受/码流变化没有验证。

McBufferInfo.pts在输出收集后没有用于划分/转发每张图像的时间；多个dequeue的data拼同out，发送仍一个Sample+固定frame_duration并Default其他字段。不能据此称每输入图像的PTS已完整传到浏览器，也不能无运行证据断言某次必拼接多个不同picture。编码队列空/无output导致画面缺失、采样时间与state推进的关系须实际测，不以固定fps参数证明。

init/state、三任务/容量2、capture丢帧/重复上一帧、增强完成后才try_send等仍与前项同类；编码改动会改变时间/压缩失真，因而不能将两变体的展示差别直接归为增强算法进步。SDP固定profile-level-id与实际MediaCodec输出SPS/能力匹配未验证；本调研不做完整协议/FFI审核或修上游。

## README、性能与许可的证据等级

英文README仍称OpenH264、无需system codec、四阶段/通道4及build.sh，和当前MediaCodec依赖/13文件包不一致；软件包不是名字所暗示的一份新神经模型。README Resource Usage自述Snapdragon835/Adreno540、720p500kbps、24fps、RSS50–80MB、视频CPU2–3核约1.5GHz、6.6MB二进制。这里保留为**作者自述**，没有测量脚本、日期/源码/所选codec、增强开关/强度、warmup、样本或原始日志，不能归为此NDK快照的独立硬编/增强基准，也不能与MobileIE神经网络ms同口径比较。

README/vnrit.md称MIT并链接LICENSE，Cargo称Apache-2.0，完整包无LICENSE；MIE-0006的授权文件不能直接补成本项已核。只记录冲突/待澄清，不作法律结论。API笔记和锁文件有版本线索，但未编译、未执行test_mc/整应用，具体ABI/格式/设备/所有依赖/运行安全仍unknown。

## 价值与后续可证伪问题

价值是展示CPU传统增强可与Android编码器接线，并提供“编码/采样/显示”不能忽略的工程线索。它不增加一个独立视频MobileIE算法先例或证明手机端时序增强已解决。可迁移为固定编码器和真实帧时间的工程对照；不能迁移的是未核的RSS/CPU/质量/硬编保证和网络模型能力。

后续需先固定共同输入、全部增强开关/状态、尺寸/精度/codec名/格式/码率/profile、真实采集/编码输出PTS与展示时间，区分预编码图像和传输后的质量/时序；对照CPU OpenH264/MediaCodec时同时核硬件实际选择、warmup/峰值RSS/能耗与掉/重帧。test_mc只能作为拟议兼容性入口线索，不替代增强时序GT与共同MobileIE控制。所有建议属于后续实验分支，不在本阶段执行。
