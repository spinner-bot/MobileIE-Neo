# R27 原body FST时间算子：实现门槛与预实验协议

创建：2026-11-12T00:54:02+08:00；作者：Codex  (sqs_A100)；原因：R26b排除固定RGB聚合主要收益后，聚焦同参数FST计算顺序，先验证可比性。

## 当前阶段：实现验证，尚不启动质量训练

仅改变body二次非线性，head、空间/全局读出及tail沿用原始4075参数。令q(u)=(w1*u)*(w2*u)+b，u是当前body.block1特征，v是上一帧未融合的同层特征，W为固定teacher输入flow+FB/bounds接受域；拒绝像素使用当前同层值。五种固定控制：

- copy：q(u)，无历史。
- post_transport：.5*(q(u)+Wq(v))；缓存上一帧q(v)。
- post_after_warp：.5*(q(u)+q(Wv))；缓存v。
- pre_square：q(.5*(u+Wv))；缓存v。
- cross：(w1*u)*(w2*Wv)+b；缓存v。

无新参数、alpha或辅助损失。第一帧/reset/current替代直接q(u)。流式状态只存上一帧未融合当前特征的detach clone，不存最终融合body；训练双输入保留过去编码器梯度。这是有限一帧历史，非无限递归。12通道缓存/带宽/特征warp开销需报告，零新参数不代表零成本。跨帧双线性先例已读ABM，算子及平方展开不当作新颖性。

## 事先固定的实现门槛

CPU先验证，无DEV质量读取/更新：OpenCV INTER_LINEAR的1/32网格、replicate边界、原始坐标bounds及FBmask。随机RGB/12通道、整数及分数平移、拒绝mask，与OpenCV参考FP32最大误差<=1e-3（归一化特征测试用输入*255后比较）；FP64算子恒等式相对尺度误差<=1e-12。检验

    post_after_warp − pre_square = a/4*(u−Wv)^2
    pre_square − cross = a/4*(u−Wv)^2
    post_transport − post_after_warp = a/2*(W(v^2)−(Wv)^2)

a=w1*w2；只在共同接受像素成立。此空间插值矩含结构及噪声，不是真实时间噪声方差证明。自动求导gradcheck验证特征值，flow/mask不求导。

五模型参数数目4075；CPU FP32当前极限/首帧/全拒绝fallback相对原始CausalFST plain最大归一化输出差<=2e-5；同一冻结权重直接双输入和流式第二帧差<=2e-5。缓存应恰好12*H*W*4字节且reset释放，copy为0；应能证明非copy过去输入梯度非零，copy不依赖过去，过去扰动实际影响causal输出。全部失败保留，不能仅放宽门槛。

CPU门槛过后才安排GPU同样验证，仍每次两次空闲采样/同组最多四卡。需检查CUDA gather反向确定性：启用torch.use_deterministic_algorithms(True)，进程局部CUBLAS_WORKSPACE_CONFIG=:4096:8，不改共享环境。若不支持，保留失败后先决定算子实现，不能静默切过去stopgrad或混算子。

训练预算/所有强参考/数据恒等审计将在通过实现门槛后单独补充并先提交；本版只授权CPU/GPU实现验证，不以冻结替换初始质量判定训练后成败。最终保留素材不读。

元数据校正：2026-11-12T00:56:01.335441+08:00；原手写创建时间超前，改用首次协议提交时间；不改测试门槛/算子/实验范围，CPU启动配置仍保留原协议SHA与Git版本可追溯。

## 2026-11-12T00:58:05.818944+08:00 原训练流字段门槛

在CPU两seed固定step索引0/1/3999，核对R26原current/GT/ROI/合成metadata；新字段准备与原align_tasks在同CPU后端的4项history SHA和RGB数组必须完全一致。共6组/24历史，不把CPU teacher与旧CUDA teacher数值混为精确重放。公开fields重采样RGB与原aligned RGB归一化差*255<=1e-3。copy的forward/损失/梯度/一次Adam相对新建同CPU CausalFST plain逐tensor完全一致；6次copy更新仅实现验证，不留作质量候选。只读训练来源，不读最终保留素材。CUDA同样验证待空闲卡及算子门槛通过后再执行。

## 2026-11-12T01:02:26.746229+08:00 冻结原生序列算子探针（CPU可独立执行）

CPU前向/stream/current门槛已过，copy一次Adam首测失败未定位，新增诊断六组全部精确但不能抹去首败；CUDA仍等待空闲。本节仅冻结前向，不启动正式质量训练，也不依赖未通过的优化器门槛。

固定5套4075参数权重：官方、R26 plain Charb双seed4000（与R14一致）、R26 plain MSE双seed4000；原四native DEV/16帧、关闭reset，五body算子各current/causal，共200逻辑配置。所有current以及copy causal应与同权重原plain前向/指标精确一致并作显式alias；每权重/片段六套实际输出（五算子causal、plain finite post .1/.5，其中copy同原plain，合计七套，140指标计算/60精确current额外别名口径需按下文明确）。

确切计数：5权重×4片×(5算子×2history+2经典post)=240逻辑行；其中20个copy current和80个其它current均alias到copy causal，共100alias；实际神经推理100序列（五causal算子），另40经典post仅warp/blend，指标140套。所有字段来源固定输入teacher flow/mask，评价用原reference flow/mask，不读最终保留素材。模型冻结，绝不从GT计算部署flow；当前极限/首帧归一化差<=2e-5，逐帧及逐片alias精确，既存CPU plain参考可匹配时另核验，不混CUDA数字。

每配置保存完整时序/质量指标和输入/权重SHA，报告双目标、双seed、每片，不把冻结替换初始冲击当训练后失败，也不按结果删模式。目的在真实长宽、实际fields下核验与比较算子行为，非新方法/SOTA证据。训练预算仍待实现/优化器门槛充分处理后单独预注册。
