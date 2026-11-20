# MIE-0009｜reef-track 阅读过程

Codex  (sqs_A100) <langxibielangle@qq.com>；2026-11-20。固定公开源码静态分析，0上游导入/pytest/模型/视频运行、训练、GPU或Gate3实验。

## 身份与材料

固定[tediyankov/reef-track](https://github.com/tediyankov/reef-track/tree/af76a927c568795e800f8d8e0c28c3b939fd6f1f) revision af76a927c568795e800f8d8e0c28c3b939fd6f1f，GitHub元数据2026-11-10T11:24:37Z；pyproject作者Teddy Yankov/version0.1.0，账号不等于MobileIE论文作者，也不据此确定初次公开日。核心是水下鱼类GUI交互跟踪、3D礁体重建及行为标注，MobileIE是可选显示/跟踪输入增强。

完整215278B ZIP含87文件/81初始可读文本；原Gate1增强/README/GUI三个codehit与当前包对应摘要全相同。81链接只是准备，不是全部语义精读。额外提取.gitattributes和LICENSE-MobileIE；前者仅行尾规则，没有以export-ignore隐藏NPZ。

完整阅读README、enhance模块、环境YAML/pyproject/.gitignore/.gitattributes、tests/test_enhance.py、tracking/roi.py；适用阅读GUI初始化/可用性/播放/渲染/选择/送SAM、TrackingSession主接口/preview/传播、video探测/FrameMap/proxy命令/reader主体与devices；methods文档0–1节和总体架构、配置相关项、data说明。另完整重核原MobileIE的MobileIELLENetS/FSTS适用原代码。不声称全GUI/所有tracking、3D重建/融合/同步/安全/完整依赖审计。没有取得真实GoPro输入/GT/增强输出或科学图表，README字符架构实际读过，不当视觉质量验证。

## 这次是直接算子移植，不只是名称

[结构/权重核对](model-and-weight-audit.json)逐项对照原作[MIE-0002模型](../../../MIE-0002/derived/docs/upstream-code/model--lle.py.txt)与[FSTS](../../../MIE-0002/derived/docs/upstream-code/model--utils.py.txt)：head Conv5(pad2)→PReLU→Conv3(pad1)→FST、body Conv3→FST、空间平均→1×1卷积→sigmoid、通道max→1×1卷积→sigmoid、x3*x2*x1→tail Conv3。FST仍是(weight1*y)*(weight2*y)+bias；没有误折掉非线性，也没有额外图像残差。torch.max与amax此处是取相同轴的最大值拓扑对应，不宣称不同设备/核数值逐位一致。

输入从uint8 RGB变NCHW float/255，输出clamp0…1、乘255转uint8再回CPU NumPy，范围约定符合原RGB路径。实例只保留权重/device；没有历史图像/H/C/flow/时间loss/新视频训练。GUI cache共享实例不是时间记忆，tracking worker make创建另一个实例也没有改变这一点。

**不能从静态拓扑对应推出已验证权重忠实性。** 文档/模块称UIEB转换npz已在enhance目录，固定87文件里实际没有mobileie_uieb.npz；.gitignore排除全部*.npz，pyproject package-data通配只声明打包意图，不生成文件。固定contents路径实际404，当前作者公开releases实际[]，见[原始尝试](weight-entry-attempts.json)。后者Nov20结果不证明过去从未有过release；没有识别可取得的该转换件或转换脚本。

4075是原作模型计数及本项目测试断言，不是本次对缺失NPZ的实物计数；键/形状/数值/原checkpoint hash/UIEB训练来源均未核。原MobileIE资料已取得并不能替代本项目缺失转换件，也不自行造权重填补作者缺口。

## 可用性、测试与默认

available(mobileie)同时要求find_spec(torch)和NPZ.exists；当前固定包缺NPZ，安装PyTorch本身不满足可用。GUI把不可用项禁用，tooltip只说needsPyTorch，不能据此把缺权重误归单纯依赖环境。annotator轻环境不含Torch，完整环境给Torch>=2.6等范围而非锁定组合；没有在共享/项目环境安装这些外部依赖。

普通显示默认Off（可读取既存用户偏好），SAM sees enhanced默认勾选仅在选择增强时生效，不等于默认执行MobileIE。MobileIE优先MPS→CUDA→CPU，和通用SAM/recon device解析CUDA→MPS→CPU分别记录；未执行设备选择。无MobileIE autocast/half显式路径。

tests/test_enhance.py全文只作为作者测试设计阅读：参数化按available过滤，专用MobileIE测试又skipif缺可用性；若按固定包缺NPZ条件，MobileIE覆盖不会因总体pytest成功而自动成立。其断言为4075参数、同图确定性、与输入有变化，并非与官方实现对齐/GT空间质量/视频闪烁/持续时延。未运行这些测试，不声称现环境测试通过。

## 视频、ROI、时间与信息条件

GUI full view开启增强且宽度>ENHANCE_W960时先缩小再增强；zoom时先原尺寸crop增强再缩放显示。SAM的TrackingSession._view独立make，先crop再增强，无此显示宽960上限；因此full-view显示与SAM可能不是相同分辨率输入，即使选同一算法。增强选项记录到segment，原视频不被该view覆盖；proxy创建/压缩是另一流程。

ROI类FishFollower有中心smooth0.5、模板0.8/0.2更新和逐帧位置cache，用于观察窗口稳定/跟鱼，不是MobileIE像素时间稳定。跟踪ROI每帧按前mask重心移动；global平均注意力与Retinex逐crop percentile会受内容/边界变化影响，这是需控制的潜在表观不稳定来源，尚未测幅度/是否可见。

MobileIE增强本身每次只处理当前提供图像；**整套跟踪不能称在线纯因果视频增强**：默认backfill=True，按人工点击/修正可双向传播，SAM还有跨帧记忆；这些不属于增强器时间状态。预览窗口还可能使用离线已跟踪mask，须固定ROI/人工信息才可作公平时序测试。appearance检查仍用原始RGB，SAM可用增强，不能把其追踪指标直接当图像增强GT。

FrameMap保存start+stride*i及标称source_fps，119.88/目标30例stride4，实际proxy29.97；proxy ffmpeg命令重新生成CFR PTS。对已解码帧号可追溯，不等于取得真实每帧相机PTS/VFR验证。FrameReader做BGR→RGB且默认cache64帧，缓存/解码成本不是网络参数；它也不是神经时间建模。

## 成本、演示与数据边界

README写增强约2–25ms/frame、30fps；代码只给QTimer设播放间隔、逐次goto/render，没有取得真实逐帧性能日志/热身/精度/完整设备与输入表。宽960显示降采样或native crop与全1080p/4K不是同成本口径。README另表M3Pro36GB、SAM约0.6s/frame、Wat3R每32帧40s/11GB是**别的模块**作者报告，不是MobileIE手机时延/参数/峰值内存。不能拿4K权重数或UI目标FPS当全pipeline实时保证。

真实GoPro示例/共享RFS/输出data均不在Git，data/README明说仅本地生成；没有访问示例中他人目录，也没有运行代理/跟踪/重建。未识别MobileIE增强前后视频、GT、时间指标或跨种子实验结果。颜色/运动/噪声表现和与SAM的实际收益均unknown。

LICENSE-MobileIE确有Apache2.0文本，作为该组件的许可声明保留；缺转换件的权重来源/授权对应尚不能由文件名认证。项目根未见单独LICENSE、pyproject未给总体license字段，不能把组件许可推为全部第三方/数据授权；不作法律结论。SAM/Wat3R等属上下游依赖语境，不递归扩成新的MobileIE相关方法身份。

## 科研价值与可证伪控制

它比纯启发LUT/BYOM标签更直接，证明公开工程已有MobileIE slim算子的**逐帧视频应用路线**；没有提供新的学得时间模块或对闪烁的解决证据。可借鉴frame编号/proxy来源记录、固定增强模式的segment配置、裁剪/显示/SAM输入分层；不能继承未得转换件的结果、4K低全链成本或新颖性。

后续实验须先有转换件/原权重哈希和算子/数值对账，再比较固定全帧/固定ROI/随动ROI、增强开关及强时间对照；分预编码图像质量、时序稳定、SAM下游分割与GUI显示，不以任一替代其他。所有方法共用输入/曝光/codec/帧时间和因果信息，记录完整解码/转移/模型/显示成本。这里只作Gate3候选，不实施。
