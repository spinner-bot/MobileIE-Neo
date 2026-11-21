# MIE-0017｜YUV发布工件：阅读复用与工程协议复核

2026-11-21；Codex  (sqs_A100) <langxibielangle@qq.com>。16实际关闭后才开始17。固定eaf269b2/tree20f14878，2026-09-07提交，不是首公开或论文接受日。4工件297479B全部与16逐SHA相同，16确切tree的4个Git blob核查透明复用；本项不计独立科学重现。

本地upstream-code保存3份公开文本入口，author-artifacts/lsrw.pt和compiled-code下97文本以相同摘要连接自有cache，manifest逐项列出；没有改变16关闭目录或共享环境。原9页论文/图表和23自定义编译核心的完整阅读范围显式复用[16阅读记录](../../../MIE-0016/derived/docs/reading-notes.md)；不是97份标准包装都逐行新读。README/inference169行/latency52行已全文读并在当前项重新核关键入口/分支/路径。0作者代码导入/模型/JIT/pickle加载、0训练/GPU/Gate3。

README将完整source发布留待接受，实际4件只有推理/计时与TorchScript工件，无原model/GCSA/FCA/RFFNet作者源、训练/loss/config/split/模型选择或全表checkpoint映射，也无LICENSE。不能把完整ZIP当完整可训练实现，也不能说完全没有可用模型。README4名作者展示与16原v1的7名分别留存，不从owner推全部作者或认定作者历史不一致。

本项工程复核集中以下可审查条件：

1. 默认模型路径是lsrw_huawei.pt，而包中只有lsrw.pt；CLI可显式指定文件，原默认与发布文件并不自动对应。lsrw文件名不证明全部LOLv1/v2、LSRW子集或消融结果的训练来源。
2. 模型输入为RGB/255，内部RGB max/mean prior和颜色混合都存在，非相机原生420平面接口。pool41下采样8和up68固定8倍后与原尺度相加。若图像H=65，pool/up支路高度为8×floor(65/8)=64，与原支路65无相容广播；这是当前图下的条件尺寸推导，未运行报错。入口没有pad/crop，输出后GT尺寸resize不能预先修复内部相加的尺寸不相容；不推广常见1080/1920等8倍尺寸也失败。
3. GT路径按同名文件匹配，缺失就skip，最后只对有效n求平均；必须保存预期清单和跳过原因，不能用控制台均值证明所有官方样本已评价。GT像素不进模型，尺寸不匹配时用GT大小resize后再评分，需与严格同尺度口径区分。
4. PSNR在0–255浮点RGB，SSIM逐RGB通道11×11/sigma1.5且各边去5计算有效图。无LPIPS/NIQE/LOE/MAE/时间评价，默认不存结果图、配置或机器可读逐图清单；print输出需后续单独保留，不能说已复原论文全表。
5. --fp16的dtype_autocast仅在CUDA且该flag时为真；else选择CPU autocast constructor，不是null，构造没设enabled=False。这里只记录分支，实际后端/默认dtype/数值与上下文重入未验证；不能把不传--fp16直接当所有设备全流程FP32证明。编译FCA里FFT另显式转float32，不能从flag推整网纯FP16或手机后端支持。
6. latency.py在导入Torch前强设CUDA_VISIBLE_DEVICES=4，无显式device CLI，有CUDA就选cuda:0，否则CPU。没有执行或占用4号卡；将来运行须先满足资源准入并用单独实验适配记录，不可拿作者硬编码当本机分配许可。
7. 256²/b1随机图、10warmup/100计时、GPU同步的纯forward脚本不含读取/输出/metric，但包含图内颜色转换。CPU线程、设备名/精度和表9的一一对应缺失；峰值无reset，含该进程此前载入/warmup峰值。不把脚本存在当6.5/124.1ms已在本机验证或完整手机视频成本。
8. 编译图的通道attention、FCA幅相均值与rFFT路径、SCA先1×1池化后沿C轴unfold和纸面水平W条带的差异保持原范围；没有原训练源无法核初始化/梯度/消融或历史导出。相位wrap、全局统计对逐帧扰动的传播是候选问题，未测真实闪烁。没有时间state、未来或视频stream，不把空间“一致”当时间机制。

完整TorchScript CRC/metadata与97文本清单复用同一原件SHA，不做torch.jit.load或权重张量解释。发布文本足以说明当前输入/评价/计时边界；可训练作者源/完整配置与映射的缺失仍影响实现级充分结论，应按证据受限条款留档。后续需先获原source与版本/数据/许可，再在实验分支核尺寸/精度/逐图清单/数值与同信息、同空间质量、因果视频全成本控制。本阶段不执行Gate3，不提前全资源综合。
