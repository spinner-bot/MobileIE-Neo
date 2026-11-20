# T2原生递归实现资格：固定两训练批，零优化更新

2026-11-20；Codex  (sqs_A100)。先于新实现执行登记，承接question.md/note72。仅回答原DIS最终RGB递归的前向及固定field条件下时间反传是否实现正确，不检验质量提升，不启动正式训练或新DEV。旧T1/V1/全部STOP保持。

唯一CPU seed0，原MixedPool step8000/8001、B4T4/128、2SYN+2真实、原排除。仅加载原seed0 F0（4075参数、完整原权重），每批16图一次有梯度前向并clip[0,1]，共32模型图/8窗口；没有教师网络/GT估计flow/新模块/优化器/权重更新/DEV/最终/GPU。FP32、CPU2/OpenCV1、TF32关闭、固定deterministic设置及自有环境沿用；实现与本协议先冻结后执行。

每窗独立原FixedDIS与.5递归、首帧raw、后3帧每次只用本模型前/当前raw估计原native双向field/FB接受，24双向对/48calc。复用R83 StateOperator的原OpenCV1/32相位/原valid支持构建Torch线性RGB传输。flow与mask stop-gradient，窗口内RGB融合历史不detach；不称整个DIS可微，不引用GT构造field。增强net仍逐帧提取，部署函数含真实过去状态。

每批再以同已估field驱动原RecursivePipeline，比较完整4帧RGB输出，**不重估field**，最大归一差≤2e-6；首帧与本raw必须恒等、输出shape/range/finite正确。对末帧GT误差分别求rec与raw的raw张量梯度：原raw路径的过去梯度须0，rec路径的过去三帧总绝对梯度须>0且全部finite；仅证明图中有过去依赖，不证明历史改善或估计flow正确。另对全窗clip-GT和递归GT两个均方loss各求一次参数梯度，所有4075参数梯度存在/finite。两批合4次raw梯度求值/4次参数梯度求值，0SGD，不做Adam逐bit或探针网格。

记录窗口ID/来源、原权重来源、源/协议、24field接受比例、两个loss、梯度总量、原生前向差和运行终态。通过条件为上述全部实现断言；通过只允许另登记完整训练/评价/源与必要CPU/CUDA资格，不能据此宣称新方法或自动打开最终。失败保留原终态，先审查是否实现错误，不自动重跑或扩mask/相位/阈值网格。没有墙钟截止，较长等待按实测sleep并核原PID/start/runtime。
