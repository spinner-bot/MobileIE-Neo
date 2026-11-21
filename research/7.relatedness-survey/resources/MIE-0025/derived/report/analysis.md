# MIE-0025｜GenAI4E/QATIE：可审查的量化工程参考，端侧与视频结论仍待验证

2026-11-21；Codex  (sqs_A100) <langxibielangle@qq.com>。有效阅读序28，继MIE-0024论文冻结结项后开始。同家族资料只计一条科学证据链。

当前作者项目完整提供了单图训练、FX QAT、ONNX/TF转换、TFLite推理及照片评测入口。其可迁移价值是帮助二期固定低精度部署协议；已有代码并未补齐视频质量、时序稳定性或手机端到端证据。入口名、文件名及CSV标签都不能代替对实际数据流和后端的核查。

## 来源与阅读边界

[官方仓库固定6703a6d](https://github.com/GenAI4E/QATIE/tree/6703a6d123042601a465378a74bdfe224696fe7d)，tree ac4bf360ee5dcf636915f3a11486d14c003dc878，head提交2026-06-04。论文精确链接与README作者/正式引用吻合；组织owner不等论文作者身份。原始37工件29145635B与固定ZIP/Gitblob及MIE24逐一SHA一致，26适用文本的完整阅读按摘要透明复用，另重读项目两README及关键入口边界。详见[来源](../../sources/manifest.json)、[原件](../docs/author-source-manifest.json)、[复用证据与14份前项审计](../docs/source-correspondence-audit.json)、[项目笔记](../docs/reading-notes.md)。

MIT许可已读；不由其推断DPED/外部预训练数据的许可。requirements原UTF16LE，解码侧档完全对应，没有安装依赖。README要求Python<3.12，并已提醒onnx2tf版本变化可能影响转换。两图实际视觉审查复用同SHA；四PYC只库存，三权重只静态容器元数据，未执行作者代码/反序列化/推理。

## 当前工程事实及限制

| 环节 | 核查所得 | 后续控制 |
|---|---|---|
| 训练入口 | README写dped/，CLI默认../data/dped与../../ckpts；示例c32/loss2/limit=.01，默认c24/loss1/limit=.1，QAT强制FP32 | 显式自有路径、完整命令/seed/预算；示例不是论文历史配置 |
| 模型与loss | 单图门控UNet，35Conv/5IN，c32=915331；loss2借鉴MobileIE，MSSSIM类实际单SSIM，VGG仅loss3 | 不把骨干当MobileIE MBR；控制容量/度量与历史版本 |
| checkpoint | loader可fallback weights_only=False，多个strict=False；ablation按键/文件名推断 | 固定配置/权重身份，不能用自动加载推断历史语义；本次不调用 |
| FX量化 | 当前训练默认ConcatFP32保浮点，shell legacy默认旧图，head保FP32 | 固定observer图/legacy；不凭INT8名称判全整数 |
| TF量化 | 裸权重经ONNX/SavedModel后再PTQ，默认128测试输入校准 | 分离校准与最终评价，固定converter与工件SHA |
| 公开TFLite | 100版float NHWC入/NCHW出；大版int8 NHWC[1,1920,1080,3]入出，算子139/128 | 检查真实shape/dtype/affine，不默改横屏；静态类型不是QNN委派证据 |
| 动态/分块 | reshape非通用transpose；正overlap线性mask外端0；InstanceNorm按块统计 | 分别控制native/pad/tile及边界，不称所有路径已失败 |
| 评测/计时 | CPU TFLite的CSV行可标cuda；原图resize、PSNR/SSIM入口、full_hd传递各异 | 对齐实际入口/配对/指标/分辨率；标签不等后端 |
| 内存/成本 | file_mb除1024²实际MiB，param_mib仅parameters求和；Torch默认1088×1920 synthetic forward | 不是完整packed权重/峰值/热功耗/手机端到端；补真实设备测量 |

参数字面求和和公开TFLite结构对应前项冻结审计，而非本项独立重跑。完整CPU载体与手机FP16/HTP、100patch质量及比赛1024成本不能混作公平同条件排名；源文档本身已说明部分评测是hostCPU。

## 对科研结论的影响

QAT、空间门控、精炼与MobileIE损失均有具体先例，直接组合不足以证明创新。论文中INT8 QAT较PTQ质量更好，但仍较FP32 QAT低1.144dB/.071；c24作者score更高，loss对照SSIM也更高，FP32表头/FP16正文歧义均保留，项目结项不能解除反证。没有跨输出state、未来帧或时序评价，单帧因果并不保证无闪烁。

历史训练命令/seed、原数据ID/许可/对齐、表与权重/手机文件SHA、逐图/trial结果、delegate覆盖、峰值内存、热功耗及真实视频表现仍未知。当前公开载体全部适用材料已分析，没有另一个宣称公开但实际未取得的独立补充内容；这支持文献/静态分析通过，不是原论文复现完成。

只向下一阶段Gate3交接三个可证伪控制：同权重跨载体范围/布局/边界等价；独立校准集的同预算浮点/PTQ/QAT；真实运动/噪声/曝光下空间质量与时间/感知指标和完整手机成本。须先在全部资源终态后统一筛选；当前不运行预实验，不编辑作者实现或主线。

主要结论见[18项claims](claims.json)，审查见[Gate2记录](../docs/gate2-review.md)、[决定](gate2.json)与[机械证据核验](../docs/conclusion-evidence-check.json)。通过关闭本项后才可进入冻结下一项。0作者源码执行、0模型加载/推理/训练/GPU/Gate3。
