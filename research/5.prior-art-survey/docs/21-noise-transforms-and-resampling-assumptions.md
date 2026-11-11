# 噪声变换与重采样的假设边界

创建：2026-11-11T22:40:00+08:00；作者：Codex  (sqs_A100)；原因：补齐14号文档未精读的Noise2VST，防止把R22空间插值矩误当可部署噪声估计。

[Noise2VST ICCV2025作者正文v2](https://arxiv.org/html/2407.17399v2) §3–4已读：以单调样条变换/偏差修正逆变换配合固定Gaussian去噪器，借blind-spot自监督拟合；零均值和空间独立是关键假设，正文明确讨论demosaic相关噪声的限制。130为变换可训练参数，不能当完整部署参数量。

[官方实现固定版本](https://github.com/sherbret/Noise2VST/blob/02b839dae08e6304441ef9e5b3452262c36c4550/src/noise2vst/models/noise2vst.py)已读样条、fit及forward：默认2000步、64方patch、batch4、Adam.01、每1/3预算衰减；masked拟合与完整可见推理不同。[官方README](https://github.com/sherbret/Noise2VST/blob/02b839dae08e6304441ef9e5b3452262c36c4550/README.md)另要求同时处理图像共享噪声分布。未运行模型，未下载权重；[源码/阅读范围记录](../process/noise2vst-readiness/source-reading.json)。

## 本项目的数学分析（非该论文结论）

令观测z=s+e，E[e]=0、Cov(e)=diag(v)，W为固定线性插值矩阵。则：

```text
Cov(W e) = W diag(v) Wᵀ
Var((W e)_j) = Σ_i W_ji² v_i
Cov((W e)_j,(W e)_k) = Σ_i W_ji W_ki v_i
```

两个目标像素共享源邻域时，即便原噪声独立，重采样后也可能相关。若flow本身由有噪输入估计，W依赖e，上述固定W条件式不能直接当无条件证明；JPEG、clipping、非线性特征又进一步改变假设。

R22的V=W(u²)-(Wu)²是一次实现的空间邻域矩。对归一化权重w和独立零均值局部噪声：

```text
E[V] = Σ_i w_i (s_i - Σ_k w_k s_k)² + Σ_i w_i v_i - Σ_i w_i² v_i
```

它混合真实结构与噪声贡献；没有额外信息时不能由V唯一恢复噪声方差。因而不将R22矩直接作为噪声标签，也不把HH-MAD或输入时间残差当真实σ。二次FST中aV的精确交换关系仍然成立，但“校正它提高最终视频质量”是另外一个必须实验检验的问题。

这也限制后续自监督设计：blind-spot/Noise2Noise的独立性不能在warp后自动继承；由同一有噪输入产生teacher/flow时尤其要区分条件选择偏差。这里没有将Noise2VST的图像结果迁移为本项目视频性能结论，未确定最终方法。
