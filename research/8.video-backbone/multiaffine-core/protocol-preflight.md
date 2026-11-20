# M1骨干原型资格：来源约束、流式实现和有限梯度

2026-11-20；Codex  (sqs_A100)。先登记、实现、冻结，再首次运行。依据[N1后问题](../triple-variance/next-core-question.md)及[note76直接先例](../../5.prior-art-survey/docs/76-polynomial-networks-and-observation-ancestry.md)。不是正式方法/质量预算，不启动GPU/DEV/最终，不用资格替代空间恢复证据。

唯一固定新骨干：A=Conv5 3→16、stride2、padding2；B=Conv3 16→16、dilation2/padding2；C=Conv3 16→12、padding1后pixelshuffle2。两个组合G_i(u,v)=.5(u+v)+gamma_i*u*v+b_i，gamma_i/b_i均逐通道16参数，初始化0。总5340参数，卷积MAC为5232×H/2×W/2=1308HW，1080p2.712269GMAC；这不是实测端侧，另有乘法/状态/移动/启动额外计算。

空间diagonal映射：z=A(x)，c=B(G1(z,z))，输出PS(C(G2(c,c)))。四槽按旧到新X0..X3：c01=B(G1(A(X0),A(X1)))、c23=B(G1(A(X2),A(X3)))，输出PS(C(G2(c23,c01)))。来源重叠控制将前一分支改为c12，故共享X2；只作资格反例，不作为事后选择的新方法。没有PReLU、norm、sigmoid/max、flow或内部clip。

流式：每帧算一个z_t和c_t=B(G1(z_t,z_{t−1}))，缓存z_{t−1}及前两c。t≥3用c_t及c_{t−2}，两帧对不相交；前三帧输出当前diagonal，额外空间body调用须计入启动成本。所有缓存独立contiguous clone/detach，不把临时frame view保留为大storage；B1 FP32稳态3×16×H/2×W/2×4=48HW字节，128图786432B、1080p99532800B，约为经典RGB状态两倍。这里不声称状态小于强post或解决运动，输入尺寸限定偶数。

CPU2/OpenCV1/FP32训练、TF32off；结构检查使用独立FP64复制，固定随机16×16 fixture和非零gamma(.4/.3)，不训练该复制。不做逐bitAdam。固定6步stream对前三当前diagonal/后三显式four槽比较，最大绝对差≤2e−10；稳态来源集合必须恰是当前四帧、两因子不交，底层缓存字节正确。固定四个整幅独立Rademacher扰动，共16个等概率角点：原raw四槽输出平均对均值输入输出差≤2e−10；同角点重叠控制应有>1e−8的期望缺陷。输出clip后的差只描述、不要求也过精确性质。此有限Bernoulli检查不是对任意噪声/实际运动的实验验证。

有限学习资格唯一原MixedPool seed0/step8000 B4T4/128，两SYN两ND，原排除/自有资产。原F0 seed0冻结教师、一次16图得到clip[0,1]目标；独立随机M1与独立随机原MobileIELLENetS(12)各两次Adam1e−3、(.9,.999)、eps1e−8/wd0/globalclip1，MSE使用未clip学生输出对教师。两学生总4丢弃更新/64学生图，16教师图另列；不以两次loss变动作效果。全部参数gradient存在/finite、更新后参数finite；保存两次loss/梯度范数和来源，所有更新丢弃，不导出供正式训练。

随机学生初始化分别使用torch seed8930/8931，结构fixture种子89300；教师仍原完整4075参数/官方预训+项目8000继承，不让随机学生冒充同预训练预算。M1不函数继承原F0，不把5340与4075参数/MAC差隐藏。实现资格全部通过只允许另登记空间表达/任务恢复实验及完整GPU资格，不能自动进入视频训练或宣称均值/画质收益。实现失败保留原记录后审查，不改科学门槛循环重试；不设墙钟截止，按实测速率sleep核原句柄。全部旧STOP/最终/原84值/PDF020/7号/其他worktree保持，小步提交/阶段推送。
