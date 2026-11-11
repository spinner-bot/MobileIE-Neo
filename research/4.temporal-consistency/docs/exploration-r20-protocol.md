# R20 动态视频的同流场聚合顺序控制

创建：2026-11-11T21:45:35.731518+08:00；作者：Codex  (sqs_A100)；原因：R19静态可用信息存在但前置不统一胜出，先核验真实运动中的强经典基线，再决定新训练。

零新训练，固定R14 native plain双seed4000、原46探索条件，保留未用test/Kodak。统一原MobileIE teacher半尺度FB flow/mask；不使用GT流部署。八种固定方法：raw；finite preblend alpha.1/.5；finite post-output-warp alpha.1/.5；finite post-aligned-input alpha.1/.5；既有recursive half01。alpha不按dev重选，全部报告。

令A=对齐后的前帧低光输入，O=原逐帧增强clamp输出，W(Oprev)=同场/mask对齐前帧输出并在拒绝像素回退Ocur。pre为F(x+alpha*(A-x))；post-output-warp为Ocur+alpha*(W(Oprev)-Ocur)；post-aligned-input为Ocur+alpha*(clamp(F(A))-Ocur)。A在拒绝像素回退当前输入；神经算子的邻域/GAP可能将该区域周围变化传播，记录此顺序差异，不称三者完全相同有效域。三类均有限两帧、不递归；recursive half01直接复用原fuse代码，是不同状态对照。

逐帧模型FP32/batch1、cuDNN deterministic/TF32同R14，raw/recursive指标与R14原逐帧/逐片结果核对（最大绝对误差≤1e-10）；不得借batch或预处理变动获得收益。边界0直接复用raw，避免加权同帧的舍入差；input-reset在事件帧将finite结果替换raw，后续有限历史不额外延长reset，recursive按原代码清状态。所有reset开关保留，raw及无事件的结果可以证明等价alias，逐帧精确审核。

八方法×双seed×46×两reset=1472逻辑行，预计真实指标1072、alias400；记录真实模型序列/帧前向数量和三类模型输入，不当1472独立视频或部署成本。共享比较会复用raw/aligned预测，不能把共享总耗时分摊成单方法速度；若记录神经阶段wall，明确不含teacher/flow/数据/指标/传输，不作全pipeline/手机结论。各方法实际部署需要的模型遍数、3通道历史及共同teacher/flow成本要分开解释。

先CPU fixture核验仿射点算子三种有限聚合等价、平方点算子Jensen差、first/reset的精确行为。每seed独立GPU任务，启动前两次空闲检查，同组≤4。数据/权重逐SHA，原始值/失败/过程/报告分文件提交，不设墙钟截断。若未形成独立动态质量—稳定性优势，继续作为强基线，不宣称新模型或算子顺序创新。
