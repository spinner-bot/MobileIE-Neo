# 数据域边界与已发表增强方法准备度

创建：2026-11-11T20:23:49.027314+08:00；作者：Codex  (sqs_A100)；原因：补齐跨数据源采集边界、官方权重与架构核查。

## SDSD：另一个相机/高ISO域，不直接等同自然夜景

实际阅读作者PDF第3–4页§3.1/3.2及§4开头；未目视渲染图，不声称全文精读。[作者PDF](https://www.cse.cuhk.edu.hk/~byu/papers/C131-ICCV2021-SDSD.pdf)记载Canon6D MarkII、重复电动滑轨路径，第二次加ND滤镜并提高ISO；手工匹配匀速运动阶段。150对、70室内80室外，原1920×1080，100–300帧。作者所述机械路径误差≤1mm不等于像素级GT误差界。§4模型以中心五帧输入，需两帧未来。

因此拟用它核验跨相机/高ISO条件；自然夜景、非刚体动态、自由手持仍是另需验证的边界（本项目判断）。不得把它包装成解决OLiVES ND域局限的自然夜景测试。

CVF403、作者大PDF普通下载IncompleteRead均记录。作者服务器支持206/ETag，使用65个256KiB分块、四CPU线程，逐Content-Range/ETag/SHA验证并重试两次实际timeout，完整16970584字节，SHA256 20b85aa9e79395c23322370ed2bf594ea007ff2672004ed1fe336a6d93e4dfbd。pypdf6.1.1只安装到项目忽略cache/工具目录，未修改共享环境；源码和过程另存。

[官方仓库](https://github.com/JIA-Lab-research/SDSD)README blob fde9b3f30b7e4b898800c42177dfb04b594299f2：原PNG与处理npy不同，官方train/test YAML需固定。公共Drive数据根和权重页面HTTP200，仅检查目录元数据；原PNG的indoor/outdoor子目录、npy子目录均能列出。尚未取得像素或选择子集，先核验split再登记取样协议。公共页面可达不保证所有大文件可完整下载。

## StableLLVE：已完成真实推理入口

[官方固定commit](https://github.com/zkawfanx/StableLLVE/tree/4cc070da328ffadc8498285985aa98175e473bc5)的model/test/dataloader/train/LICENSE已读，README明确公开checkpoint为synthetic clean，噪声生成代码当时标为待更新。U-Net推理逐帧零未来；训练L1和warp一致性不等同推理有时间状态。官方BGR/float64除255后转float32，模型内部处理奇数大小，适配器不额外resize/pad。

权重17274710字节、Git blob f819f8daf6f9ad7e2548480ae2584935f4922a45、SHA256 277e4d29566709a31f8f25c401396e9bd9c70395e6da36513cd9d9a83e005ad9；源码逐字节/许可保留。四尺寸官方路径最大差0，4316259参数，R15完成46输入/184逻辑行。它是clean权重跨域迁移，不能与任务内微调作公平论文排名，更不能由迁移失败否定原方法。详见R15协议/报告。

## FastLLVE：正文架构与可获取性

实际阅读[正文§3、§4.1–4.2及附录A/Table3](https://arxiv.org/html/2308.06749)。五层3D编码器、3D解码器生成强度图，视频条件权重融合基LUT，4D查表插值；可选3D去噪。训练含重建、LUT平滑/单调约束。故“低维映射+空间层级+视频上下文/去噪”已有明确先例，不能作为本项目新颖性。未来帧/分块边界须读官方代码后再定。[官方仓库](https://github.com/Wenhao-Li-777/FastLLVE)README blob 6b1c1b4fc8e38b6bf52954249a3ed8332b60aaac；CUDA扩展尚未构建，未运行或宣称复现。公共模型Drive元数据已列出indoor/outdoor/smid权重，未下载二进制。

下一步仅选择少数有证据的问题：强空间控制下，多尺度表征是否使真实历史获得独立收益；是否需要分离亮度映射和去噪通路。普通U-Net、3D卷积、LUT及残差组合本身不构成创新，需可证伪机制和成本验证。
