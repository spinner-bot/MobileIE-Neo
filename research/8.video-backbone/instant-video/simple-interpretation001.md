# I1-S全层/输出比较的解释边界

2026-11-21T16:28:55.507116+08:00；Codex  (sqs_A100)。在首次I1-S DEV像素/指标之前补充，触发原因是逐层复核实际MobileIELLENetS.forward、作者完整层配置及已完成CPU资格的状态清单。只收紧因果解释，原协议/模型/80K预算/44门/判定数值不改，不重跑资格或停止训练。

原Slim直接return self.tail(x4)，tail是最后Conv2d，返回之后无额外非线性。作者配置给每个Conv2d与root各加稳定器，所以all有5个12通道内部稳定器（其中att.1为1×1全局）＋tail与root两个3通道RGB稳定器。CPU实际清单也包含tail-stabilizer与root stabilizer。all新增60＋3＋3=66参数，output仅root的3参数；二者不是只差内部5个节点。

对固定已训练权重的同一RGB通道，令z_t为tail未稳定输出，则末端为

```text
u_t = beta_tail*z_t + (1-beta_tail)*u_(t-1)
y_t = beta_root*u_t + (1-beta_root)*y_(t-1)
```

两个EMA间没有clip或非线性，构成已知两级因果IIR。若禁用内部5个稳定器，z_t即原F0逐帧未clip输出，末端仍有两级RGB平滑；output只有一级。相同首帧初始化下，围绕首帧常量的变化响应是两个一阶传递函数之积。这是基础滤波代数，不是新定理，也不从该式预测PSNR或视频感知。缓存的已clip F0不能直接冒充上述未clip z_t来做精确干预。

因此all/output的4个预定增量比较只能支持“完整已发表堆叠相对单输出层级的增量”，不能独立归因于内部特征历史、FSTS节点或状态表示；额外RGB滤波阶数、参数/状态容量均有差别。现有JSON字段all_internal_increment_passed和kind=internal_increment保留为历史协议ID，**通过亦不是纯内部机制证据**。原all的20＋4门、output的20门及允许下一成本/感知的条件不放宽。

当前只是已发表基线任务适配，不主张该结构原创。若未来论文需要内部特征作用的机制结论，须另行事前登记匹配RGB级联/预算的控制或明确同权重干预的范围；同权重禁用控制也不等于重新优化后最优RGB基线。现在不据此增加训练臂、数据评价或alpha/深度搜索，不把潜在控制当已执行结果。

原数值/评价冻结保持，独立最终/原84值/PDF020/所有旧STOP不改；论文目标active未完成。
