# O1共享核心：实现与CPU学习资格

登记：2026-11-21；Codex  (sqs_A100)。原因：B1未有独立历史增量；本轮改变共享主干学习路径和非局部状态响应，不救其网格。[先例/旧STOP区别](../../5.prior-art-survey/docs/78-state-response-supervision-and-shared-core-boundaries.md)。本版先于O1代码及任何数值执行，只允许CPU资格，不预授正式训练或DEV/手机。

## 固定原型

当前空间路径完整继承各seed原4075参数F0，不替换卷积/平方/注意力。第一次平方前12通道z同时进入原读出与新描述：4倍均值池化、Conv3 12→8/PReLU8/DWConv3 8→8，padding1；无新独立RGB encoder。描述在quarter/其再4倍池化的sixteenth网格分别作逐像素L2归一eps1e−6。coarse固定半径4全二维相关、softmax后的位移期望×16；上采样到quarter后warp上一描述，再固定半径2相关预测剩余位移×4，最终场上采样native。边界无效匹配屏蔽、含零位移，温度exp(logtemp)限[1,100]、初始10，两级共用。

confidence的Conv1输入为8通道当前/粗warp描述绝对差、2通道quarter单位总位移、coarse/fine各最大相关，共12→1，末层全0。native alpha=.5*sigmoid(logit)*bounds，初始.25，不声称任意历史上的初始完整视频函数等于F0。首帧直接当前clip；后续融合自己的已clip RGB历史并clip，下一状态保存当前紧凑quarter描述与融合RGB，原低光/RGB不作额外历史。每段reset，无未来/GT启动。原当前函数初始化保持；后续所有原主干和新头可学习，不能称一直冻结或零推理开销。

预计新增974、总5049参数，以实测张量计数为准。解析卷积为(3828+59.25)HW+144 MAC；不含相关、归一、softmax、resize/warp/复制，不能当手机时延。B1FP32持久状态4*(3HW+8*ceil(H/4)*ceil(W/4))字节；1080p29030400B，仅历史非峰值。

## 两个相同初始化的监督控制

每seed matching初始化torch seed9020+seed，从同一个原型deepcopy为coordinate与response；F0空间目标为同seed冻结原网络的未clip输出，估计教师half-FB仅使用其clip输出。两组都可学习全部原主干和同一新增头，fresh Adam：原主干lr1e−5、新参数lr1e−3，betas(.9,.999)、eps1e−8、wd0、全局grad clip1。共同空间项100*MSE(raw,F0raw)，不读取亮GT作为本资格训练target。

coordinate：native教师接纳域内mean|flow−teacher_flow|/16，加全像素BCEWithLogits(logit,teacher_mask)。response：预测状态线性部分A_theta(h)=alpha*B_flow(h)，与教师A_T(h)=.5*mask*B_teacher(h)在两张低通随机状态上比较，MSE除教师响应能量下限1e−4，再对两状态平均；不额外加坐标/BCE项。教师沿原OpenCV 1/32网格及原bounds，学生连续双线性，训练/实际部署的学生采样一致。两目标尺度/梯度不同，不能仅凭同预算将结果纯归为信息因果。

每seed fixed MixedPool step8000、B4T4/128：2SYN/2ND，只用输入和来源标记，不将配对GT用于目标/推理。冻结F0共32教师图、24双向half-FB对。两随机状态以numpy RNG SeedSequence[902,seed]生成独立Gaussian三通道、分别replicate均值池化kernel5/17，每样本每通道减均值/RMS标准化，源状态48张三通道；固定在本批资格两步重复使用，非白噪声算子全空间保证。两组各两次Adam×两seed=8丢弃更新、128学生训练图，不导出资格权重/optimizer、不按资格loss排名选设计。

## 必须通过的实际资格

1. 两组初始化state_dict完全相同；首个训练前raw对原F0归一化最大误差≤2e−6，参数实数计数与协议一致。两监督都对共享head与新描述有非零有限梯度；允许初始空间项的其它层梯度为0，不假称所有参数首步非零。两步后参数/输出/损失有限，原教师权重不变。
2. 全24实际教师对、两随机状态的quantized teacher A与原OpenCV warp/bounds最大差≤1e−5；零状态响应严格0。它是传输实现资格，不是估计flow是真值。拒绝/越界回退当前的完整T与A的区别写清；不以错误的fallback给A加偏置。
3. 每个两步后的模型在该批第一个四帧窗比较完整stream与无缓存functional执行：各4张×两执行×四模型=32额外学生图，两者归一最大差≤2e−6；首帧等于该模型当前clip输出，实际历史替代会改变后续输出。stream只拥有两个独立紧凑状态，字节数符合本尺寸公式，不持有完整clip storage。差异门槛是实际非零>1e−8，不能以状态对象存在替代历史生效。

总学生图预期160（128学习＋32流式一致性），另32冻结教师；其内部配对匹配/状态响应调用单列，不当成160独立样本或推理计时。不做CPU逐bit Adam、额外数据/seed/超参数网格。所有CPU参数和资格输出留档，失败原样保留，不放宽数值界或自动重试科学配置；实现错误须另记修正及新增尝试，原源/结果不覆盖。

通过只能支持另登记完整学习预算与空间/独立历史/坐标和强post完整比较，不能直接称方法有效、新颖或手机可行。失败停止此固定共享描述/两级soft相关/响应状态组合的局部调参，不重开R74/R88或B1/M1网格。主线main、自有data、指定作者/no-ff、文件级密集提交/阶段推送、原84值/PDF020/所有独立最终/7号和其他worktree保持，论文goal active。
