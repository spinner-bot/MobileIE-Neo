# L1固定CPU实现与学习通路资格

登记：2026-11-21T10:49:30.134782+08:00；Codex  (sqs_A100)。先于本原型首次真实窗口前向/更新；当前仅授权资格，不自动启动CUDA/正式训练/DEV。普通隐状态与GRU已有先例，见[问题](question.md)和[note79](../../5.prior-art-survey/docs/79-latent-state-recurrence-and-transport-boundaries.md)。O1及所有旧STOP不解封。

## 唯一实现

原各seed R44 square_mse唯一最终4075权重，经既有load_spatial只读加载后deepcopy，当前原head/body/attention/tail全部共同可学。保留原FSTS顺序乘法。原head第一次平方前z先avgpool4/stride4/ceil，再逐位置对12通道LayerNorm(eps1e−5、可学scale/bias)、Conv1 12→32、tanh得到e。previous=e当首帧或current路线，否则上一state。

cat(e,previous)→DWConv3 dilation4/padding4/groups64→Conv1 64→64→sigmoid，沿通道得reset r与update u。cat(e,r*previous)→DWConv3 dilation8/padding8/groups64→Conv1 64→32→tanh得到candidate q；s=(1−u)*previous+u*q。所有DW/point卷积有bias，只有最后readout Conv1 32→12权重/bias全零，其余使用Torch默认初始化；new seed9100+seed。readout(s)连续bilinear/align_corners=False上采样到z并相加，再进原FSTS和后续原网络。新状态是生成的潜在特征，非噪声方差或物理运动；其更新虽有凸门，但candidate不是原始样本。

总12431参数；新增8356含LayerNorm24，新增Conv kernel8064。解析Conv为3828HW+8064ceil(H/4)ceil(W/4)+144，1080p8.982835344G。B1 FP32持久state=128ceil(H/4)ceil(W/4)字节，128²为131072、1080p16588800；不含归一化/门/激活/pool/resize/clone或workspace，不当完整RAM/手机速度。tanh e/q及sigmoid凸门在有限实数运算给s∈[−1,1]；不推出整个输出/Jacobian收缩或长期画质稳定。

同seed两个完全同初值的独立模型：spatial每帧previous=e，causal接前state。spatial执行同cell/readout，不跳支路；全部参数实际存在，但初始readout0会使前部梯度0，须第二次丢弃更新确认打开。current同权重干预是逐帧重新以e作previous，不是假装位移干预或独立训练空间模型。stream保存仅一个detach contiguous clone state，显式reset序列。

## 一次CPU资格

CUDA_VISIBLE_DEVICES=''，CPU2/OpenCV1、FP32、确定性，os.nice5。两个seed各MixedPool(seed,step8000) B4T4/128，2 SYN+2 ND，只有既有57 DAVIS train和12实拍train，不读DEV/最终。GT仅资格训练MSE目标，模型接口只接低光和自己的state，无GT/flow/teacher部署输入。原F0仅生成一次未clip初始对照，不用它作这次学习目标。

每seed两个模型各两次fresh Adam丢弃更新，共8；原参数lr1e−5/新增1e−3，betas(.9,.999)/eps1e−8/wd0/globalclip1，目标为全部四帧未clipRGB对GT的MSE，无额外时间loss或坐标教师。不能继承这些权重/Adam，不能按loss选设计。首次前向每模型与同seed原F0最大差≤1e−5；两个初值state_dict逐tensor相同。两次的loss、输出、活动参数、梯度均有限且都有梯度tensor，第二次共享head/project/gates/candidate/readout的norm须各>0。

更新后每模型用第一条四帧low：functional与逐帧stream最大差≤2e−6；state finite、absmax≤1+1e−6，compact独占storage恰131072B。current与causal模型各自按自己训练路线stream；另跑同权重current，第一帧相同。末帧输出对前三帧low梯度：causal绝对和>1e−8，spatial严格0；causal末帧与current末帧最大差>1e−8。再reset读取第一帧11×15低光裁块，输出shape原样/有限、state=(1,32,3,4)/1536B。只核这些固定输入和路径，不作所有尺寸数值/无穷长稳定保证。

预算：32原F0图；196学生图=128学习图+16functional+16stream+16current+16过去梯度+4奇数尺寸；8丢弃更新、0流/正式优化/GPU/新DEV。完整窗口来源、梯度与状态检查和终态保留，不导出资格权重。任何失败原样停止，不更改cell/宽度/dilation/初始/目标/LR/种子或容差救资格；技术原因需先保留并独立判断，不能把旧任务重启。

具名源码冻结只覆盖登记清单，不宣称整个传递闭包；实现/协议/冻结/结果/过程/报告分别密集提交。全过只允许另写固定正式学习及完整质量—历史—强post/成本协议，不能据此称已有视频方法或创新。research/O1/指定作者/no-ff、自有data、原84/PDF020/独立最终/7号/并行main与其他worktree保持。
