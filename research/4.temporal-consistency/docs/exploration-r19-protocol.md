# R19 已知时间信息与噪声相关性诊断

创建：2026-11-11T21:24:00.078190+08:00；作者：Codex  (sqs_A100)；原因：R16空间层级与R18整体配对均未解释独立历史收益小，先用已知完美对应核验模型的信息利用。

仅使用已知训练Kodak01/06，ImageOps.fit768×512后固定[y192:320,x320:448]128×128；两photon20/80、JPEG off/on70、rho0/.9，共16条件、每条件64独立观测对。静态同一GT，一过去零未来、已知零运动，不需要GT光流；不把静态无运动收益当真实视频结果，不读最终保留素材。

低光c=(GT/255)^2.2*.35，current count~Poisson(c*p)，current read~N(0,.008^2)。给定current count N，shared~Binomial(N,rho)，history count=shared+Poisson((1-rho)c*p)；history read=rho*current_read+sqrt(1-rho^2)*independent_read。两帧边际噪声形式一致、前处理相关系数rho；之后同旧degrade截断/uint8量化和可选JPEG70，实际相关性另测，不能把rho当最终观测相关。current RNG1901900000+scene_index*100000+photons*100、history RNG1902900000+scene_index*100000+photons*100+rho_index，相同current跨rho逐像素完全一致，相同传感器样本跨JPEG条件一致；新RNG不是旧R6像素重放。

CPU fixture先核验rho1编码后两帧精确相同、跨rho current不变、preclip边际均值/方差/协方差的固定大样本误差（相对方差/相关≤.02，均值绝对≤.0002），以及总体Var((X+Y)/2)=(VarX+VarY+2Cov)/4恒等式。所有统计以固定像素跨64draw总体矩FP64计算，记录实际input方差/协方差/相关及平均方差比；不是对真实传感器noise label的估计。

固定模型：official原始；R14 native plain双seed4000；R16 spatial/raw_history/aligned_history各双seed4000，共9权重。只读SHA验证，不再训练。所有模型报告current-only与前置等权平均(current+history)/2后current-only；R16两个时间类另报告真实history（静态直接history输入，不执行部署flow）。同权重current替代保留，禁止只比较训练后整体均值。共22模型/路径×16=352RGB统计行，所有候选全报。

推理FP32、batch8、cuDNN deterministic/TF32同前；同轮所有路径相同batch，不与旧batch直接宣称像素等价。输出先记clip比例再clamp[0,1]；margin8，记录条件MSE=均值偏差平方+draw方差、noise SD255、mean draw PSNR、每draw MSE和closure（绝对≤1e-12）。前置平均是传统静态参考，不是新方法或免费实际运动处理；若有效仅说明在此受控条件下有可用时间信息，不能证明真实运动/曝光场景均有效。

一项GPU作业仅空闲同组卡经两次预检启动，缓存/权重忽略、源码/检查/结果/过程/报告分文件密集提交。按算法64对、全部16条件完成，无墙钟截断；按阶段SSH推送。若某项门槛失败留档，不扩大阈值或换更好素材。
