# R63 保留 dense 原函数的联合状态：先行资格检查

创建：2026-11-13T01:06:02.055948+08:00；作者：Codex  (sqs_A100)。本登记只授权下列无图像资料、无优化器、无CUDA的CPU结构检查；不登记或启动质量训练，不打开开发/最终像素。目的为区分已有负结果中尚未隔离的因果问题，不把普通状态模块作为新方法。

## 问题与已知边界

R34把原dense body换成depthwise+pointwise，同时改变空间函数；R37保留dense但冻结全部骨干，而且current绕过全部新参数；R48共同训练的是RGB输出递归。它们不共同隔离“保留原dense函数、内部状态与原骨干共同训练、同容量空间支路有效”的条件。R5/TSM、R6去偏、R27跨帧二次、R45读出门控等已有结果继续冻结。本次直接复用R33的固定状态路径，不能将重复组件改名为创新。

## 唯一结构及空间对照

沿用R33 pre-head-FST特征u（12通道），pool4、12→8投影、[current,previous]16通道depthwise3×3和pointwise16→8/PReLU；三通道field提供±2低分辨率像素offset与sigmoid置信度；越界退回current。s=current+confidence*(warp(previous)-current)，8→12读回并双线性上采样加到u。readout初始全零、field初始全零。唯一改变是**保留原始dense body及其余全部空间计算**；继承R44 square_mse每seed8000累计更新的expanded MBR训练权重，骨干共同可训练（原未用tail_warm保持冻结）。无新增宽度/尺度/融合系数网格。

同初始化三组：plain原expanded骨干；spatial保留完整新增路径但每帧previous=current，field/采样/读回实际执行；causal上一递归状态。spatial不是跳过路径的current消融，且offset仍可形成空间变形；不要求其各参数在所有输入上均可辨识。首帧以current代替missing history。causal的同权重current是诊断，不能替代单独共同训练的spatial对照。全参数学习为未来可能训练设定，本轮0优化器步。

## 固定CPU检查与证伪标准

只使用torch随机张量，两seed、两组B×T×3×H×W（1×4×3×17×19及2×4×3×16×20），CPU2线程。记录来源checkpoint及本轮源码摘要；不做Adam逐bit replay或精度网格。

1. train和eval中，零readout的plain/spatial/causal输出对齐，max_abs≤1e-5；expanded与部署fold eval≤1e-4。与plain同样每帧调用一次BN，不把train-mode expanded和slim作错误等价比较。
2. 零readout时旧骨干和readout收到有限非零任务梯度；上游新投影/field初始零梯度是零读回的正常结果，应显式报告，不伪称所有路径即刻解锁。
3. 丢弃的**非训练资格扰动**只对readout加固定随机1e-3权重（不优化、不保存为候选）。验证此时causal有真实历史输入梯度，spatial对其它帧梯度严格0；spatial自身新路径投影/field/readout及骨干存在有限非零梯度。record按模块列范数，任何缺梯度/非有限/所需非零失败则整体资格失败。大于0只证明计算通路，不证明可学收益。
4. 扰动后因果首帧与current一致；同一参数/相同前缀而任意更换未来时，前缀输出不变；causal与current在某个非首帧有非零差异。若不成立停止。
5. 扰动后expanded→slim完整序列≤1e-4；fold仅合并空间MBR，历史支路继续存在，不能声称折回单帧或零推理开销。导出参数必须逐项记录并保留原dense12→12的3×3核；新增状态严格8×ceil(H/4)×ceil(W/4)×元素字节数，step尺寸变化重置。也记录1080p/720p理论持久状态字节，不当峰值内存或时延。

异常/失败原样保留，不自动重试/降低容差；本登记不含真实GPU/训练/画质/速度结果。CPU资格通过后只能说明该比较可实施；是否值得正式训练仍须独立登记固定数据与预算、双seed、plain/spatial/causal、强自身raw递归参考、画质—时间门槛及真实成本。若不能回答比前轮更明确的问题，则不启动训练。完成后源码、资格结果、解释、过程、进展报告分别小步提交，阶段推送；所有大体量资产留zrh自有data。
