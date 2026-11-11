# R11 强优化下的原始历史与对应几何

创建：2026-11-11T18:45:22.655202+08:00；作者：Codex  (sqs_A100)；原因：R9/R10确认原低LR基线不足、粗平滑损伤质量；在更强优化下重检可学习历史窗口。

## 固定训练

三种4975参数模型：copy_high（R8 spatial_copy）、concat_high（原始前帧）、concat_aligned_high（同前帧先按固定官方teacher半尺度Farneback FB对齐）。各seed0/1、1000更新，共6000步。主干与900参数分支统一Adam1e-3、wd0，所有参数共同clip1；与R9full1e-3强单帧基线比较。R9full1e-3在新dev/旧合成较强，但不称统一最优，本轮探索选择明确依赖前轮dev结果。单一学习率不证明整个结构族最优。

保持R5每步当前/GT混合身份和Charbonnier .001，无新增时序loss/训练素材/更新预算。所有分支零初始化；初始输出/未clip主干梯度匹配plain。global梯度norm包含辅助分支，可与plain不同，不声称首步clip后更新完全等价。

对齐训练对每对原始低光输入使用冻结官方teacher（batch1、clippedRGB），半尺度FB输出当前→前帧field，cv2双线性gather原始低光前帧，bounds与FB交集有效，否则填当前输入；第一帧始终自身。真实pair可按scene/t缓存，合成每step各自退化后估计，不能用GT或已知合成运动替代。CPU flow与teacher额外成本完整记录，无免费运动声明。

## 固定评价

46条件：原38探索、4项目dev resize480270、相同4dev的R10 native_crop512288。后者与前者同物理源，不是新增独立视频。每个temporal权重causal/current历史，copy只有causal；全部input reset开关、raw/共同half01 .1。另重评R9full1e-3两个权重同46条件/两reset/两fusion。新结果共(1+2+2+1)×2seed×46×2reset×2fusion=2208行，其中plain368、copy368、concat736、aligned736。

对齐causal使用缓存的同官方teacher半尺度FB；current历史为严格当前输入，reset首帧也严格当前，不能让infer通用reset抹掉current设置。输出half01使用同motion，独立统计额外输出融合收益。既有42条件的plain无reset/raw+half01应重放R9；原生4条件重放R10，预设最大指标差1e-6。记录PSNR/SSIM/edge/GT残差时间误差、逐片/逐帧、first-frame与事件reset，不能只宣布PSNR增益。

## 意义与限制

目标是分开强优化、同参数空间与历史对应因素，先建立可复验的有效窗口；拼接、光流和FB均已有先例，不当创新。若效果仍不统一，保留负结果并收缩具体候选，不推断所有视频架构无效。后续正式强视频模型、长训练、自然夜景和端到端/手机成本仍必需。

每GPU任务新两次空闲预检，计划同组2/3，最多4空闲卡；无墙钟截止，不覆盖失败。源码、测试、训练、评价、过程、报告分别小步提交，cache像素/权重只记SHA。
