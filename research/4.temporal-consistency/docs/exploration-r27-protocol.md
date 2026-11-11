# R27 原body FST时间算子：实现门槛与预实验协议

创建：2026-11-12T00:57:00+08:00；作者：Codex  (sqs_A100)；原因：R26b排除固定RGB聚合主要收益后，聚焦同参数FST计算顺序，先验证可比性。

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
