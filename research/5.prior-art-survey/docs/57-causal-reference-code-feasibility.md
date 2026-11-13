# 因果RGB参考：EVRNet来源边界与ReBotNet发布代码资格

2026-11-13；Codex  (sqs_A100)。R74完整止损后的定向可用性核查，接续[48号阅读](48-state-information-initialization-and-device-comparisons.md)，不扩大自定义模块网格。

## EVRNet暂不纳入官方代码复现

[作者v1](https://arxiv.org/html/2012.02228v1)§3.1规定当前RGB、前帧RGB和2通道latent，经alignment/differential/fusion恢复当前帧与新状态。§4.1规定卷积后的PReLU及50epoch约50K更新；因此不能把短预算、无激活的同名实现当充分复现。[作者Amit Kumar出版页](https://akumarcv.github.io/)和[Sachin Mehta主页](https://sacmehta.github.io/)确认ACM MM2021接受记录，但本次相关条目只确认论文链接，没有确认作者代码入口。GitHub名称检索无官方结果不等于证明全网未发布。

唯一进一步读到的[第三方URTSR移植](https://github.com/BrandnerKasper/URTSR/blob/9f48cc259ac429be89b83235be87fc19d39ece89/SRcode/models/evrnet.py)服务于游戏超分硕士项目：3通道随机CUDA隐状态、每次detach、缺少论文PReLU，PixelShuffle后固定8输入通道，只有scale2满足其32通道布局。不是可以直接设scale1的等分辨率增强模型。detach是否不同于原作者训练并未由论文文本证实；不能用第三方选择反推原论文。未下载/运行/重新分发该移植；根目录树未出现LICENSE，不扩成全仓法律结论。EVRNet继续只作结构先例，不填复现质量表。

## ReBotNet有作者认证的架构代码，但不是完整可运行基准

[作者项目页Code](https://jeya-maria-jose.github.io/rebotnet-web/)交叉链接[作者仓库](https://github.com/jeya-maria-jose/rebot-net)，固定提交`dff81d3cbd484b01ef2d42da2dd1ad1332598a09`。该树提供CC0许可、架构、XS/S/M/L配置和PortraitVideo退化加载器；没有训练器、递归评价器、权重或完整数据工具。仅架构/配置/许可进入`third_party/rebotnet`；不调用不完整加载器，不获取新数据。

实际源代码约束：

- `archs.py`需要`B,2,3,H,W`，第1项为历史预测，第2项为当前帧并作残差基底；官方`configs.py`例子却传4D随机张量，不能照抄其计时输出。
- 两个mixer均硬编码576 token，四次stride2和16像素分块形成固定384×384工作尺寸。改变构造参数`img_size`不会改变576；任意原生分辨率、128训练裁块不能直接套用。矩形同token数量是否可运行需实际检查，不能视作已经支持任意大小。
- 发布XS为深度[4,3,3,4]、通道[8,12,20,32]、mixer hidden64、depth4；这是发布配置，不是已核对最终WACV论文的等价表。
- 架构本身是一个两帧函数，未包含递归初始化或时间反传循环。自启动、历史detach、展开长度由适配入口决定，必须写明。
- [arXiv v1](https://arxiv.org/html/2303.13504v1)§3.4/4.3使用GT/高质量首帧。我们的部署条件没有该信息；任何退化首帧自启动结果必须注明适配，不能称原表复现。WACV正式全文访问仍受限，不将v1和发布代码自动归为最终稿一致。

决定：先用原架构、CPU、人工输入做一次有界功能资格：384有效输入、错误4D/128失败边界、跨预测历史梯度、参数和Conv/Linear MAC。0新视频像素、0SGD、0GPU、0质量行；不跑1000次延迟基准。若可用，后续须另定原生分辨率分块/重叠融合、训练展开、预算和强递归对照协议；完整训练计算与初始化差异不能靠相同步数消除。普通基准适配不作创新；不因找到代码自动训练或开启旧STOP。

## 原代码CPU资格实测完成

[qualification001](../process/causal-reference-code/cpu-qualification001.json)正常结束，3.361秒、两CPU线程、CUDA未初始化。384方形及256×576同token矩形有效；原4D例子、128方形、320×640矩形均在预期位置失败。因此准确限制是固定token数量和各级尺寸约束，不能绝对说只支持384方形。两个连续预测的历史梯度非零，仅证明可反传，不证明时间收益。注册890275参数；一个384两帧调用的Conv/Linear MAC为784862208，未计norm/激活/加法/拷贝/分块融合，不外推成手机时延。GPU没有被占用。

环境仅在自有data下新增可选导入依赖；Torch仍2.9.1+cu126，vision0.24.1+cu126、timm1.0.22等版本及真实导入路径已记录，原架构不作替换。原文v1另确认8×A100、500000更新、Adam初始4e-4/cosine至1e-7及Charbonnier；绝不能把本次3秒功能检查或以后短预算适配当充分训练的原表结果。完整自启动原生分块训练参考仍待独立协议与实测资源资格。
