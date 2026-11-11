# R5c：补齐 post 位置的单帧容量对照

创建：2026-11-11T16:27:24.297281+08:00；作者：Codex  (sqs_A100)；原因：已完成 R5 的 temporal_post 在真实域 raw 比 plain 高约 .677 dB，但同权重 causal−current 仅 −.00115 dB，不能归因历史。

新增 spatial_post：与 temporal_post 完全同位、同108参数，DW处理当前 ordinary FST 的4倍池化特征，在FST后相加，不保存历史。原R5的 spatial_bilinear 位于乘积内部，不能单独充当 post 的严格位置匹配容量对照。

固定原R5任务损失、real12_mix数据、seed0/1和1000更新预算；共2次、2000更新。初始化与主干初始梯度应与 plain 相同。原R5证据、checkpoint及启动源码hash保持可追溯，此为基于已看结果的探索补充，不称预先独立验证。

评价 raw/half01 的固定38条件，每种子76行，共152行。比较 spatial_post 与 temporal_post 的训练后效果，及 temporal_post 同权重 current/off；不以单次均值定方向。测量单帧新增计算，没有状态不能把时间缓存成本算给它。若需要优化时序移位的实现，必须另列优化实现和数值等价检查，不能混用旧训练/评价成本记录。
