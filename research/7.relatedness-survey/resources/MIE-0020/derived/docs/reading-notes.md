# MIE-0020｜MobileIE-6Ch逐模块静态阅读

2026-11-21；Codex  (sqs_A100) <langxibielangle@qq.com>。19结项后才开始本项。固定66bd821a98210db4d1a84ee719ecd452f9c8295f/tree45844cf83b6424f7b6b319636285fb48a4ef426a，提交日期2026-06-02T07:05:41Z，不当首次公开或报告发表日。

## 资料与实际阅读

35件597296B，原ZIP CRC及Git官方确切tree的35个blob逐一匹配。27个可读文本准备并留存；中英文README、模型卡、复现说明、infer、配置/launcher/options、数据/损失/模型/folding、DDP主入口全506行、metrics全301行、logger及适用模块已静态读。utils_IWO全368行读，utils全文件以完整diff对照已读IWO核其差异；继承ISP/LLE四输出路径也读，但不是当前推荐6输出推理。CITATION/团队用于身份，模板/ignore/attributes核用途，LICENSE存在且Apache2.0首尾/声明核，不声称完成法律解释。35原件入口与所有过程在本项，未运行作者脚本、Torch/JIT/pickle反序列化、训练/GPU/Gate3。

作为本项佐证，关联挑战报告arxiv2605.02212v1的协议p1–2、HIT方法3.7/p5、p3主表和p17完整表、p15队员身份选择性读取；3/5/17实际渲染查看。不是开始/全文完成队列中MIE-0031（分析序35），其它团队未在此逐项分析。PyTorch v2.5.1 DistributedSampler原源仅作API语义佐证，全文静态读、不导入，不新增科学资源身份。

## 身份、直接继承与参数数值

README/CITATION及挑战报告HIT-LLIE-team对应Xinbai Wang、Duo Liu，HIT；报告p5明确以MobileIE[83]为baseline。项目是实质派生，不是同名巧合。相对已关闭MIE2原作lle.py的完整diff显示：保留head/body FST、全局通道池化与空间通道max的双调制骨架；尾从直接RGB三输出换成六输出Retinex重建，MBR改用utils_IWO。

**6Ch是输出，不是输入**：输入单张RGB三通道；尾预测三RGB illumination与三加性residual，illum=.99sigmoid(z)+.01，y=x/illum+res。没有多帧、过去/未来帧、时间loss、状态缓存/首帧/scene-cut协议。RGB照明与残差改善的物理解释不能仅由张量命名确认；损失主要监督最终enhanced，对照明/噪声分解本身没有独立GT识别证明。

README、模型卡、复现说明及挑战表均报101922。当前文件result/model_best.pt恰好101922B；仅ZIP/CRC、pickle操作码字面size tuple核查19个FloatStorage，共95528B，对应23882标量。与当前channels32源码19个参数形状逐键完全匹配：head11746、body9282、att1056、att1 64、tail1734，总23882。没有加载或执行模型，不能称独立跑出参数量。结论限定：**本固定发布的源码/检查点字面形状支持23882，而101922是文件字节数且与报告参数主张不一致**；历史提交是否不同、报告列口径/填报过程未得，不推故意或所有旧成绩无效。

仅一个32通道1080p FP32 feature的尺寸算术即253.125MiB，4K为1012.5MiB；不是实际峰值，但足以说明小权重不保证全分辨率低内存。FST为两标量乘同feature后相乘加通道bias的平方非线性，不可把整网络当纯仿射或全部折成单Conv。illum下限.01令单独x/illum项可放大至100倍，res可能补偿，不能单凭公式称实际放大100倍噪声/闪烁。

## 推理、视觉和官方表对应

infer_6channel.py全59行：直接实例化slim channels32、默认严格load_state_dict、CPU/CUDA自动选择，cv2读8bit彩色→RGB/255，单图原尺寸forward，无缩图/分块/混合精度/计时器，输出clamp和uint8再按原扩展名保存。既没有手机部署/转换链，也不能由CPU回退证明CPU手机性能。小写扩展名glob、失败读取警告跳过；未作视频时序排序或PTS处理。

作者唯一JPG海报的6组0001/0017/0029/0072/0098/0102均实际查看：可见度提高，部分输出色调较淡；0017墙/地面有明显平坦分块/阶梯状视觉，0098阴影细节平，只有压缩海报、无GT/逐张原始预测，不作定量误差或归因。无视频材料。

挑战原文p3表1与p17表2的HIT行均为SSIM.5766/LPIPS.5176/DISTS.2319/LIQE2.2977/MUSIQ60.3387/QAlign3.2007，主表17个交报告队第7、含未交报告完整表第9；排名口径已核，不能任选一名次泛称全赛。协议p2约束**提交模型文件小于1MB**，并非手机时延测试；349训练/49验证/102测试、3024×4032、Huawei/Apple图像、测试GT不公开，都是报告描述。本项未取得官方数据和逐名划分/许可，不冒称独立评测。

报告p5描述MBR训练、800epoch+20warmup、Adam/batch16、768crop、5fold、loss和gradclip.5。这与发布代码存在以下有边界的对应缺口，不据此否定历史方案。

## 构造、折叠与默认训练

model/__init__.py构造按model_task与type选择，并由-use_6channel添加6；pretrained缺省空、need_slim false。config/lle.yaml默认**re-parameterized**，所以train_ddp.sh默认训练slim，无MBR多分支；README/报告MBR训练不能作为当前默认实际路径。

原MBR分支utils_IWO里conv_out.weight冻结、另learned weight1，forward/fold都用两者之和；其余Conv/BN分支通过pad/拼接/BN running统计合并。仅静态代数阅读，不做浮点等价或训练BN与eval等价认证。阶段FST/PReLU保留，不能无条件折叠整个网络。

slim.forward_warm返回同一enhanced两次；LossWarmup却同时对第二项向input、第一项向GT约束，非原多头暖启动。若切original，存在单独tail_warm而main forward不用它、DDP find_unused_parameters=False；warmup还直接net.module.forward_warm绕过DDP包装。只记录代码路径与待验证分布式行为，不声称已观察梯度不同步或死锁。

LossLLE对out/GT先clamp，权重1 Charbonnier+.5箱式SSIM+.1空间展平逐通道余弦+.1FFT幅度L1；不是时间、flow、噪声GT或illum GT监督。ColorLoss沿H*W做每通道余弦，不能按注释读成逐像素RGB向量夹角。最终输出监督不足以证明illum/res分解可识别。

## 数据、验证、选模及日志

LLEData初始化os.listdir未排序；create_kfold用sorted all_images算索引，再赋给另建未排序img_li。给定同目录顺序可形成互斥子集，但不保证声明的字典序fold或跨环境可复现；没逐名清单/seed和训练日志。默认current_fold0，脚本无遍历五fold或融合五模型流程，不能把配置5当已实际完成五折。

训练和valid_data使用同opt/同LLEData，没有train/eval标志，GT存在就随机crop、flip与input gamma。默认768patch、globalcrop概率.5/scale.15..6、gamma.7..1.3；故验证也增强/裁块，不是固定完整官方49验证场景。输入GT空间变换绑定，但Gamma只改输入；这对原图/补光域泛化和选模须明确。

main_ddp.py:201–220给验证也分配DistributedSampler(world_size,rank)，384–407只rank0遍历并平均，没有其它rank指标allreduce。官方PyTorch2.5.1原源确认默认sampler按rank步进分片、非整除会补索引。因此world_size>1且验证集非空时，这里的SSIM是rank0分片的随机增强验证，而非全部fold聚合。没运行断言任何特定设备的首异常；其它rank与rank0 DDP forward的实际collective交互也未测。

EMA只rank0初始化在warmup之前，warmup不update EMA，之后训练每步更新、验证apply/restore。保存按rank0验证SSIM最佳，仅state_dict；没有发布完整训练日志/最终训练选模来源。lr_scheduler配置虽写cosine，实际固定CosineAnnealingWarmRestarts(50,2,1e-7)；train.resume字段与代码读取model.resume不同；main常规训练无每20轮save_every流程，最新含optimizer的checkpoint保存只在warmup。不能据README800及配置值认证此权重训练轨迹。

FastMetrics实际训练只算RGB均方PSNR与11×11箱式含padding SSIM，lpips/liqe为0字段；FullMetrics/TopK虽定义，main未使用。FullMetrics中缺lpips/pyiqa时有固定.5/50等fallback，这些不是实测指标；不能把它们归给官方挑战服务器成绩。requirements没有显式lpips/pyiqa/tensorboard/PyYAML锁（option用yaml，logger无条件导入SummaryWriter），未安装验证，不推必失败；宽下限也不是可复现实验环境锁。

通用main test路径test_inp==test_gt（config均指competition/test/low），pretrained空，LLEData仍做训练增强且返回CPUtensor、test没主动移device；若照默认cuda直接用可能有设备/参考问题，推荐infer入口已另行正确转device/加载weight。仅条件路径问题，不说推荐infer已失败。ISP默认路径/PixelShuffle不属于本RGB6输出任务，没有isp.yaml也不能拿来验证本模型RAW功能。

## 科研价值、限制及下一步

这是明确MobileIE派生的RGB照明+残差、空间训练/注意力/重参数化先例。二期不能把六通道重建、双注意力或空间频率/颜色loss本身说成原创视频机制。可交接“低illum条件下照明与残差扰动如何分别影响时间误差”，以及“同预算direct RGB/四输出/六输出与训练/验证控制”的问题，但需要等权重/训练/因果信息、真实运动/曝光/噪声、固定数据划分和完整手机pipeline成本。此处没有开展任何实验或证明假说。

尚缺当前/历史提交身份与参数统计解释、原训练配置/seed/fold清单/日志/选择轨迹、官方数据许可及完整指标执行、TFLite/移动全成本/视频GT。不把公开源完整当历史数值可重现。结项应限定当前完整静态载体，所有历史/设备/时序unknown保留；若新合法资料出现追加复审，不改冻结结论。

派生原件中的.gitignore/.gitattributes曾直接做相对链接，Git将这些特殊文件作为规则读取时出现symlink警告；本项未结项前已改名*.source，source_path及原ZIP不变。阶段push对旧历史对象有warning但成功完成，按用户保留历史要求不重写旧提交；后续不再让派生文件名承载Git规则。
