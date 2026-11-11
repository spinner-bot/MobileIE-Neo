# R15 已发表零未来增强模型迁移参考

创建：2026-11-11T19:48:46.380383+08:00；作者：Codex  (sqs_A100)；原因：R14深度训练期间补齐一个已发表LLVE训练方法的真实推理入口，区别于R12通用视频去噪。

固定StableLLVE官方commit4cc070da328ffadc8498285985aa98175e473bc5、checkpoint.pth（17274710字节、Git blobf819f8daf6f9ad7e2548480ae2584935f4922a45），另记SHA256；MIT源码保留许可。作者README明确该权重由synthetic clean数据训练，未提供本轮噪声适配，不以失败证明论文方法无效，也不称完整噪声训练复现。

已读model/test/dataloader/train：官方cv2 BGR输入/255、BGR输出；U-Net自身处理奇数尺寸，禁止额外resize或外部pad。推理无过去/未来输入，训练时一致性损失与零时域状态需分开。直接严格加载权重，输出转换RGB后以浮点评价，不经PNG量化，不加噪声、不微调。

先检查BGR路由fixture、四种非整倍数尺寸及固定uint8输入与官方test.py数学路径：归一化输出最大差预设≤1e-6；全参数strict、输出finite。不预设跨域PSNR提升门槛，迁移失败完整保留。

原46探索/dev/native条件，raw/共同official-teacher半尺度FB alpha.1及input-reset开关，共184配置—序列行；逐帧、逐片、PSNR/SSIM/edge/GT残差时间均保留。逐帧模型的raw reset及无事件结果可alias，记录真实推理/指标次数，不称184独立视频。半尺度flow仍基于共同MobileIE teacher，增加额外推理/历史状态，不当零成本或原StableLLVE方法一部分。与已存原始/强微调MobileIE及R12对照分别报告，不以训练预算不同作创新归因。

只读既有46素材，不查看保留test与Kodak13–24。source/weight验证Git对象与bytes，权重在忽略cache，源码/检查/评价/过程/报告分别提交。R14占用2/3，计划在同组真正空闲GPU1独立预检后运行本基线，最多总4张空闲卡；不改共享Python环境，不设墙钟截断。阶段耗时若记录，明示不含CPU全流程，不声称手机端实时。
