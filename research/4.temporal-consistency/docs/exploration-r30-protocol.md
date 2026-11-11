# R30：独立公开视频来源的固定迁移探索

登记：2026-11-12 03:18 +08:00；作者：Codex  (sqs_A100)。登记时只读取了DAVIS压缩包目录和train/val文本，图像像素未读，R28 seed1仍在共享GPU等待队列。此轮不增加训练或候选算子。

## 数据与保留范围

来源为[DAVIS官方2017下载页](https://davischallenge.org/davis2017/code.html)的480p TrainVal压缩包，832766765字节。官方TRAIN共60场景、VAL共30场景且不交叠；[元数据](../process/external-davis/official-metadata.json)保留来源、ETag、原划分和ZIP成员CRC。只取官方TRAIN按字典序索引0/30/59，即bear、lucia、walking，各前64连续帧。该规则在读取图像前登记，不按内容、难度或指标换片；下载失败保留失败，不用其他场景补位。

只选择性获取这192张TRAIN JPEG，不读取VAL的图像成员。DAVIS是公开视频分割数据来源；下面是自定义合成低光迁移探索，不是官方LLVE基准，也不是自然夜景。FastDVDnet的DAVIS训练来源须在引入其权重前核对并披露，不能把TRAIN来源上的迁移比较冒充独立公开测试排名。现有保留OLiVES TEST三片与Kodak13–24也保持未读。

以原480p全帧RGB作为GT，不改变FOV或二次缩放。复用`exploration_r2_data.degrade`，光子数仅20/80，读噪声标准差.008、gamma2.2后乘.35、clip/uint8；`jpeg=False`不再额外编码JPEG，来源本身仍有JPEG压缩。每个场景/光子数用SHA256字符串`R30:{scene}:{photons}:0`的前8字节little-endian作NumPy随机种子；每帧顺序采独立泊松/读噪声。共6输入条件、3独立视频来源，不增加rho、gain、JPEG或seed网格。

## 先完成的数据准备

所有原图、运行缓存只写入自有 `.cache` 链接所指资产目录。下载核对ZIP大小/CRC，记录各源JPEG SHA256；缓存保留float32 GT/low/官方MobileIE预测及共同对应。固定官方MobileIE为部署运动teacher，half-resolution Farneback/FB/bounds作为共同输入对应；GT估计full-resolution FB只作评价，绝不进入推理。CPU准备仅用Torch两线程/OpenCV一线程，不初始化CUDA，不新增训练。

## 固定终点评价与解释

只用4000步权重：R26 plain Charb/MSE双seed、R26 pre/MSE backbone双seed、R28 post_transport Charb/MSE双seed，以及官方MobileIE。post_transport同时评价真实causal和current单帧替代；其它权重只评价单帧。每个单帧映射固定raw/finite post .5，真实causal固定raw；reset关闭/开启均保留，用原输入事件规则，不制造事件。这是15个神经映射×6输入=90段实际序列（reset可导致额外计算，按实现和事件实际计数），不将逻辑行数当独立视频数。

本轮固定共同CPU后端（共享GPU等待期间仍可推进），与既有CUDA/CPU结果分表；不据跨后端微差排序。沿用全帧float64 RGB PSNR、valid Gaussian SSIM、Sobel边缘误差和GT估计flow的残差时序误差/border8，逐片、逐输入条件、分seed报告。每个神经序列先核对有限值/首帧，actual和current必须分开，不把历史收益算作单帧部署收益。门槛仍是相对固定强post PSNR不低于-.05dB且时序至少改善10%，并同时检查SSIM/边缘；不能在看新域结果后调checkpoint、系数或选最优loss。

FastDVDnet/StableLLVE在后续共同输入的公开方法对照中接入，保留原始权重迁移/训练来源/未来帧和手机未测项；不以本轮先完成的轻模型表替代它们。真实手机域另核查LLIV-Phone原始视频访问与无参考限制，无GT时不报告伪有参考PSNR。任何本轮结果都只支持外部开发集探索，正式完整公开验证、同域训练参考、手机实测及最终保留测试仍未完成。
