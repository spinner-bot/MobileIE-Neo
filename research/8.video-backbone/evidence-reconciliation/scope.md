# 已完成证据归并：区分恢复、历史信息与成本

2026-11-21；Codex  (sqs_A100)。A1完整STOP后，先于本归并脚本执行固定分析范围。只读取已结束L1/A1的版本化逐窗指标、完成标记及抽样元数据；不解码npz、不加载模型/新数据/DEV/最终，不增加CNN/优化/flow/GPU，不重设任何旧门槛。

精确数值比较只在L1与A1共同的32唯一四帧输入抽样窗×两个原F0 seed内进行；验证三类原F0指标行一致、manifest的key/draw/domain和A1来源摘要确与L1匹配。不同主题/不同抽样或空间尺寸的数字不合并排名。两seed共享输入不能当64独立场景或置信区间。

全部14项配对（每项64窗-seed/4域seed组）：L1 F0_FB/F0、F0_DIS/F0、spatial_current/F0、spatial_FB/F0_FB、spatial_DIS/F0_DIS、causal/causal_current、causal/causal_self4、causal/spatial_FB、causal/spatial_DIS；A1 current/F0、cached/current、cached/hold、cached_FB/F0_FB、cached_DIS/F0_DIS。报告平均PSNR差、最差/最好窗、均值Etemp比、SSIM/edge差和严格PSNR/时间同时正窗数；不是新增方法通过标准。全部896配对/56统计及训练场景组成保留，不选正例，不按结果改变旧STOP。

归并回答：①空间继续训练收益是否已独立列出；②历史增量与强post差距有多大；③局部投影是否先损失空间功能；④哪些解释仍未被隔离；⑤下一研究投资还缺何种实质信息。新近文献只补先例和可获取信息边界，不把普通循环、频带、残差/关键帧或多滤波混合作为新颖性。论文84值/snapshot003/PDF020不改。
