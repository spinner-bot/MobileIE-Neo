# MIE-0025 QATIE作者项目：阅读复用与入口核查

2026-11-21；Codex  (sqs_A100) <langxibielangle@qq.com>。有效序28，先验MIE24已pass_full并冻结后才begin25。此项评价当前公开项目载体，不是第二次独立科学实验。

## 全部原件与实际阅读

固定GenAI4E/QATIE@6703a6d123042601a465378a74bdfe224696fe7d，treeac4bf360ee5dcf636915f3a11486d14c003dc878。全部37文件29145635B逐一与当前ZIP、Gitblob及MIE24原件SHA一致；26文本准备不等阅读。MIE24已实际全读的26适用文本逐SHA复用；本项另完整重读README214、src/README183，并重新核train主入口195–218、to_tflite主入口220–295、checkpoint loader313–345等边界。逐文件及14个既有审计SHA见source-correspondence-audit.json。不虚报重新独立阅读或执行全部代码。

26文本为两README、MIT、UTF16依赖、四shell、模型/损失/训练/scheduler、两data、两loading/export helpers、两batch转换、三eval/两benchmark、FXquantize/TFexport/infer。requirements以BOM解码，UTF8侧档与原始codepoint一致，全部100逻辑行在24已读；没有安装依赖。两作者PNG实际审阅透明复用24同一原件与白底预览，原件和本项预览入口均有SHA；三权重只复用冻结的CRC/操作码或官方schema元数据，绝不加载。四PYC只清单，envexample不作为认证信息使用，dped/.gitkeep不等数据已获。

## 官方性、许可与环境

论文p1链接精确仓库，README作者/正式引用吻合。固定head提交时间2026-06-04，不当首次发布日期。README News声称2026-04-24代码发布、2026-03-26录用，分别记录为作者新闻，正式版已由24独立核CVF。GitHub组织owner、提交者与论文作者不混为一人。

MIT2026GenAI4E许可全文已读；不能推导DPED、外部VGG预训练/数据的许可。README要求Python<3.12，requirements固定Torch2.11+cu126、TF2.20、LiteRT2.1.2、onnx2tf1.28.8等；未安装或检查组合运行兼容。四cpython-312 PYC的存在不能取代环境规范。README明确20/04/2026提醒onnx2tf变化可能影响提供checkpoint转换，建议重训或使用已转换文件；这是作者已披露的兼容风险，不是本次运行失败。

## 项目入口与可恢复的协议

1. root README在仓库根用python src/train/train_qat.py，但dataset说明为dped/，实际CLI默认--data_dir ../data/dped、--ckpt_dir ../../ckpts；路径相对运行cwd，未来必须显式指定自有数据/输出，不能将示例当可自动正确定位。没有启动这些命令。
2. README训练示例c32/loss2/bf16/QAT/limit=.01，CLI默认c24/loss1/limit=.1；QAT准备后强制precision32。--pretrained_ckpt与--ckpt-path不同用途，不会自动恢复论文完整两阶段、预算、seed。patch_data字符串False仍真是具体条件风险，默认True不因此失效。
3. to_tflite主入口channels默认0→从checkpoint推断，失败回32；README示例显式24只能用于匹配权重，不应无条件套提供c32权重。主入口CUDA_VISIBLE_DEVICES=-1，batch导入不等自动同一设备隔离。--input_h/w默认100，--dynamic另走wrapper。没有执行导出/修改环境变量或设备准入。
4. load_torch_checkpoint_trusted先weights_only=True、异常后False；loader的名称/文档“safe”不表示纯静态元数据解析。多个strict=False报告missing/unexpected，且无residual与base可同键，自动按键/文件名选ablation不能证历史语义。本调研只用zipfile/pickletools/自有FlatBuffer脚本，没调用此包装。
5. FX quantize重建qnnpack QAT图，head与默认ConcatFP32保FP32；legacy选项重建旧concat图，shell默认legacy=1而当前train默认新图。需固定实际checkpoint图，不能把默认切换说成已经加载失败。
6. TF路径是裸浮点权重→ONNX→SavedModel→TFLite PTQ；代表集默认128个DPED测试输入，FX的FP32observer warmup也读iPhone测试。需隔离校准/最终评价，不能称用测试GT训练。两路径不是同一量化graph，公开二进制与论文表的生成链尚无完整配置记录。

## 科学实现及三权重的复用结论

门控三尺度UNet/InstanceNorm、35Conv/5IN、全局残差clamp、P(c)=889c²+156c+3，c32=915331，对应论文参数rounding；1080×1920卷积144.156672G MAC仅为字面算术。没有MBR折叠骨干、时间state/未来帧。loss2 Y域PSNR/全RGBcos/outlier直接参考MobileIE；单层SSIM类命名MSSSIM，VGG只loss3。单帧/门控/量化先例不直接解决视频闪烁。

CKPT11342475B完整CRC和操作码含observer键、epoch/global_step等文字，但从未反序列化。TFLite100版输入FLOAT32 NHWC/输出FLOAT32 NCHW，139op；大版INT8[1,1920,1080,3] I/O、128op，静态无float typedop。signedINT8 affine范围不必与论文UINT8本质冲突；也不证明QNN委派、运行正确或速度。二者不只是尺寸差异，必须固定工件SHA/布局/范围。

动态wrapper reshape不是通用transpose；强制tile/回退且overlap>0时线性mask外端点0，存在边界零权重条件风险；InstanceNorm使块/全图统计不同。它们都不是我们观察到的模型输出或全路径失败。原图图形直接匹配的路径不由tile问题否定。

## 评价与成本：入口名称不取代实际调用

DPEDData未排序独立glob，TFLite patch loader sorted且核计数但未逐basename，全尺寸loader按basename多手机匹配；需数据清单才能证明实际配对。训练test当val与评测/校准划分必须显式说明。np.float32/ToTensor再/255正确。

PyTorch、TFLite和训练的PSNR饱和阈值不一致，SSIM8bitRGB路径也未由两个入口固定成完全同定义。TFLite评测按固定图形resize输入/Canon；benchmark所谓hostCPU与手机QNN另属不同carrier。FP32batch每模型CPU评一次，生成cpu/cuda两标签行；INT8batch按标签重复CPU，均不构成CUDA加速。src文档已经披露hostCPU，不能反向造“隐瞒”。

Torch synthetic benchmark默认1088×1920、20warmup/100runs/5trials，只forward；file_mb实际以1024²除是MiB，param_mib只sum(p.numel*p.element_size)，不含buffers/optimizer/allocator或完整packed模型/峰值内存。TFLite没有param_mib。INT8 benchmarkTorch分支没传full_hd而TFLite分支传；原图注释列手机native，但当前BlackBerry/Sony为2560×1440。逐入口/原始输出未执行，因此只记录协议差异。

## 可支持、不能支持、下一阶段控制

可以支持当前公开代码提供单图训练、QAT、两条转换和多种静态图像评价入口，以及具体可核的部署风险。不能支持已复现原论文训练/表、所有artifact同graph/同layout、包名INT8就全整数I/O、cuda标签就CUDA、文件小就低内存、或手机视频实时/时间稳定。论文QAT仍少1.144dB/.071、c24score最高、MS-SSIM对照SSIM更高及FP32/FP16标签歧义透明继承，不能因项目结项消除这些反证。

没有另承诺而未获得的外部公开supplement；历史配置/seed/原数据ID与许可/对齐、权重→表→手机SHA、原逐图/trial结果与峰值/热功耗/视频证据仍是unknown。下一实验分支只建议同权重载体范围/布局/边界控制、独立校准同预算对照、真实噪声/曝光/运动的空间与时间感知联合评价及实际手机完整pipeline。当前0执行作者代码/加载模型/推理/训练/GPU/Gate3，不修改外部实现，不把问题排查当创新或数值复现。
