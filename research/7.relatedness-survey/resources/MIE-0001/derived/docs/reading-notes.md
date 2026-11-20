# MIE-0001 阅读与核查记录

记录日期：2026-11-20。资料快照：2026-11-17。分析者：Codex  (sqs_A100)。这是原论文的阅读记录；引用官方实现用于核对论文，不代表已经启动 MIE-0002 的独立资源分析。未经执行的源码只能提供静态证据。

## 身份、版本与实际阅读范围

原研究为 Hailong Yan、Ao Li、Xiangtao Zhang、Zhe Liu、Zenglin Shi、Ce Zhu、Le Zhang 的 ICCV 2025 论文，CVF 正文印刷页 21949–21960，DOI 10.1109/ICCV51701.2025.02038；作者预印本页面标识 arXiv:2507.01838v1、2025-07-02。会议版正文和10页补充为本次方法判断的主依据，不能假定预印本与会议版逐字相同。确切会议发表日未由这些材料确定，年份已证实。

完整阅读正文12页（实质内容p1–8、致谢及参考文献p9–12）与补充10页（含参考文献）。全文提取件在本目录私有缓存链接 main-text.txt、supplement-text.txt，原始PDF及SHA在 [main-source.json](main-source.json)、[supplement-source.json](supplement-source.json) 和 [manifest](../../sources/manifest.json)。全文、图页不作为重新发布的论文正文提交；Git保留链接、来源与本次原创笔记。

实际查看两张全页接触表，并单独放大查看正文p3–8和补充p1、p3–8。下面图像判断基于实际工具显示，不是依据文本猜图。接触表用于核布局及确认参考文献页，不能代替小字精读。

## 正文逐节笔记

| 页／节 | 阅读所得与边界 |
|---|---|
| p1 摘要、§1 | 目标是图像增强的轻量质量与部署，涵盖LLE、UIE、ISP。Figure1为参数与质量的二维比较，不包含视频时序指标。不是针对视频闪烁提出的论文。 |
| p2 §1–2 | 作者强调标准部署算子、训练多分支和推理单分支；回顾图像增强及高效CNN。平方交互明确引用[51] Rewrite the Stars；重参数化引用RepVGG/DBB等。二期不能把这些既有机制本身当新贡献。 |
| p3 §3.1–3.2 | Figure2上方推理链仅单张图、卷积、PReLU、两处FST、HDPA、尾卷积；下方训练多分支、BN、拼接、1×1投影。Figure3展示MBRConv融合和冻结旧权重加增量的IWO。没有时间轴、前帧输入或运动对齐。 |
| p4 §3.2–3.5 | Eq1–2给IWO冻结旧投影权重加可学习增量。Eq3是缩放平方加偏置。Eq4–6先全局空间平均得到通道权重，再对加权特征做通道最大值取得局部响应，两者乘回特征。Figure4左为BN分布示意，右为误差损失曲线，不是视频误差分布。 |
| p5 §3.5、§4.1–4.2 | Eq7以绝对残差为起点，Eq8统计H、W，Eq9 tanh权重，Eq10加权误差；“local”不意味着公式实施了滑动空间窗口。Adam，lr1e-3，10轮1e-6 warm-up，2000轮，余弦重启。LOL v1/v2、UIEB、ZRR；PSNR/SSIM/LPIPS/SCORE是空间或效率评价。Table1单列GPU与SoC时延，FPS列没有标手机。 |
| p6 §4.2–4.3 | Tables2/3分别UIE640×480、ISP448×448；Figures5–7是静态场景裁图比较。正文说手机超过100FPS及UHD前景；给出的分辨率测试不能据此证明2K–8K视频实时。ISP质量并非所有指标第一：Table3 FourierISP PSNR21.65，高于Ours21.43。 |
| p7 §4.3 | Table4比较损失和重参数化；Figure8为卷积核差值，Figure9 KL通道相似性，Figure10训练曲线和1000轮后IWO，Figure11 Fourier幅度示意，Table5 FST变体。图示与消融支持作者的图像实验主张，不能单独证明这些模块产生或抑制视频闪烁。 |
| p8 §4.3–5 | Figure12训练损失、Figure13热图、Figure14单像素误差直方图；Table6注意力替换、Table7模块开销。Ours在Table6为4.035K，Table7为4.047K，需保留不一致。IWO/LVW为训练策略，不能据此称新增视频状态也没有推理成本。 |
| p9–12 | 致谢及完整参考文献；[98] SGZ题目包含low-light image/video（WACV2022），原文Table1已有此对照。不能声称没有轻量视频先例。本调查按已冻结的直接被引／明确派生边界收集，不扩成全部后向参考文献综述；此处SGZ作为原文背景，尚未作为本调查独立全文结论。 |

## 补充逐节与图表核查

| 页／节 | 实际核查 |
|---|---|
| p1 §1–2、Table1 | 表中推理12通道；训练分支宽48、拼接480/384/96通道。保留FST于推理路径，不是整网都合成一层线性卷积。 |
| p2 §2.1–2.2 | BN融合及多核补零、拼接、1×1压缩的推导。冻结BN统计是静态融合的条件；平方后的符号展开不能直接按逐元素平方卷积核理解。 |
| p3 §2.2–3 | Figure1潜水员及彩色伞的FST静态裁图；Eq8 HDPA梯度解释；LVW描述衣物纹理及木柄。作者说AI Benchmark及PT→ONNX→TF→TFLite流程，没有连续视频评价。 |
| p4 §3–5 | Figure2衣物/木柄损失裁图；Table2四手机与GPU；Figure3实际截图顶部明确标GPU、FP16，输入与输出均[1,3,600,400]，平均6.72ms、标准差0.75ms、初始化576ms。不能把这张截图概括成任意后端／精度、端到端相机吞吐或稳态功耗。截图H/W顺序与正文尺寸书写也需保留，而非擅自统一。限制节承认CNN成本随分辨率上升，讨论LUT/量化/剪枝/蒸馏方向。 |
| p5 Tables3–4、Figure4 | Phone2/3/4的LLE13.9/7.8/72.6ms；UIE25.1/11.4/93.6ms。两组静态LOL v1图像，能看曝光与色彩差异，不提供相邻帧稳定证据。 |
| p6 Table5、Figure5 | ISP17.5/17.2/45.1ms；LOL v2座椅和球场静态图。此处Figure5任务不同于Table5，引用必须指出具体图表而非只写页码。 |
| p7 Figure6 | 两个UIEB场景及各自RGB直方图。Ours更接近参考的色彩分布是这些示例的视觉观察，不能替代全数据定量或时序证明。 |
| p8 Figure7 | ZRR树木路牌与工程机械两组静态ISP比较；没有视频连续帧或流式RAW元数据。 |
| p6–10参考文献 | 完整阅读。未发现需按已冻结纳入边界新增的明确MobileIE后续资源；此句不表示所有后向文献已另行全文阅读。 |

## 官方源码交叉核对

固定revision：7030d125e28a0b9c30075f84f79f165acd13bdc5。从已归档官方ZIP读取，不导入、不执行外部脚本。原文件、SHA与可解析本地链接见 [upstream-code-manifest.json](upstream-code-manifest.json)。论文支持性读取19文件；utils_IWO通过完整utils.py加两者逐段diff读取，未改动原文件。LICENSE标Apache-2.0；这不自动证明数据集或每个第三方转换器具有相同许可。

1. `model/lle.py`、`model/isp.py`：分别3通道RGB和4通道packed RAW，ISP PixelShuffle(2)。只有当前输入，没有history/future/state参数。HDPA的max沿dim1，即通道；不能误读成多帧max。训练模型的slim调用逐块融合，FST仍留在推理图。
2. `model/utils.py`：MBRConv5/3/1的卷积原响应与BN响应拼接，slim用running统计融合后乘1×1压缩核。FST与FSTS都持有两个标量weight及C通道bias，执行两个缩放响应相乘。等效Scale=weight1×weight2，非线性仍在；未执行等价误差测量。
3. `model/utils_IWO.py`与普通utils差异：冻结conv_out.weight，新增Xavier初始化weight1，forward与slim使用二者之和；bias仍可训练。`model/lle.py`/`isp.py`默认import普通utils；`model/__init__.py`、`main.py`与配置未发现自动切换到IWO的第二阶段入口。因此“有IWO原语”和“默认命令完整复现论文训练”必须分开；完整切换/加载流程仍未知。
4. `loss.py:17–22`：统计的是有符号out−lab，std/sqrt(2)，tanh权重detach；不是Eq7–10绝对残差统计的逐字实现。`39–59`实际LossLLE/ISP还包括通道cosine与2×PSNRLoss。未验证这些差异对质量与时序的量级影响。
5. `main.py`、`option.py`、两个yaml：标准Adam/余弦重启；LLE2000轮、warm-up true；ISP配置1000轮且warm-up false，与正文统述不同。不能把论文2000轮直接当所有公开默认配置。
6. `data/__init__.py`、`data/lledata.py`、`data/ispdata.py`：按文件名配对，RGB/255、RAW/4095及RGGB打包；os.listdir独立图像序列、无时间窗口或视频划分防泄漏规则。未下载数据集或核对逐样本重叠；不能保证训练测试不存在泄漏。
7. `torch_to_onnx.py`固定[1,3,400,600]、opset12、无dynamic_axes；TFLite两个测试脚本逐文件推理并clip，未显式设置delegate/精度/线程。脚本不构成端侧AI Benchmark环境的完整可复现实验说明。
8. `complexity.py`引用lle.MobileIES，但该固定版本模型类为MobileIELLENet/S；其thop结果叫FLOPs亦须核MAC换算。不运行脚本来偷偷修复或把静态审查称复现。

## 静态算术、推断及反证边界

12通道LLE按源码构造计数：912（5×5）+12（PReLU）+1308（3×3）+14（FST）+1308（3×3）+14（FST）+156（global1×1）+24（local1×1）+327（tail）=4075。ISP首层多300、末端少243，合4132。若去掉两处FST的28参数，分别4047/4104，与论文部分表值吻合；这是对账线索，**未证实作者统计工具具体遗漏原因**。Table6还差12，不强行解释。

补充Eq7若被理解成“普通卷积核逐元素平方后卷积输入平方”，会丢交叉项：K=[1,1]，x=[1,−1]，(K·x)²=0，而(K²)·(x²)=2。这是符号解释的反例，不是运行模型的预实验，也不能在未澄清作者记号前断言整项方法错误。更可靠的源码事实是平方算子实际保留。

输入扰动经FST有a[(z+δ)²−z²]=2azδ+aδ²；全局池化注意力可让局部扰动影响全局权重。这只能提出闪烁敏感性假设，不能证明实际视频闪烁由哪个模块造成。相反，真实光照变化本应引起输出变化，过强时间平滑也可能造成拖影／曝光滞后。需运动与曝光控制反证。

1080p RGB FP32一帧历史的纯张量算术为1920×1080×3×4=24,883,200B≈23.73MiB，不含特征、工作区、对齐和框架开销；不是设备实测RAM。任何时序方案的评估必须包含这些新增项。

本阶段0模型执行、0训练、0GPU实验、0设备测速；观察图表、PDF渲染、源码静态计数不称模型复现。下一阶段尚未获任务，不启动Gate3。
