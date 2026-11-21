# A1 决定：固定局部仿射缓存配置停止

2026-11-21；Codex  (sqs_A100)。按[原协议](protocol.md)执行唯一固定配置；[全部48均值/20门](results/screen001/tables.md)、[终态](process/terminal001.json)、[512逐窗配对/32统计](report/analysis001/paired-summary.json)与[实际图审](report/analysis001/actual-review001.json)为依据。必要门0/20，其中表示保护0/4、强组合0/16，完整False；不登记实际成本，不扩训练/DEV/JOD/外部/手机。

32唯一四帧训练场景抽样窗按两原F0 seed重复，共64窗/768输出/3072逐帧；实际新CNN/优化/GPU均0，512次逻辑F0请求不能叫新增CNN或部署速度。缓存原三基线共192指标行、五字段对原L1最大差0；GT跨seed字节相同，96个GT评价FB对，1152个新经典部署双向对。人工资格先于真实像素通过，但不是方法成功。原3899949/start117884588正常54.768514秒、tool49264退出0，原PID已无，21具名源不变。

|域/seed|current−F0 PSNR|current/F0 Etemp|cached−current PSNR|cached/current Etemp|cached_FB−F0_FB PSNR|cached_FB/F0_FB Etemp|
|---|---:|---:|---:|---:|---:|---:|
|SYN/0|−2.806242|1.396550|−.364632|1.064035|−2.269550|1.429724|
|ND/0|−.109846|.901918|−.193344|1.027354|−.380872|.990247|
|SYN/1|−2.731463|1.411038|−.317584|1.063741|−2.156548|1.427532|
|ND/1|−.254287|.910529|−.048518|1.028745|−.352995|.997484|

cached_DIS相对原F0_DIS：SYN两个seed质量−2.487305/−2.412249dB、时间比1.463390/1.477131；ND质量−.324990/−.309046dB、时间比.972130/.980248。ND部分时间下降是真实取舍，但均值/最差质量不达保护；不能据较低时间误称通过。每帧current空间投影在SYN已严重损失，因此本次不支持“先保质量，再靠隔帧复用省整链成本”。隔帧与current差异也混合映射变化/输入噪声/运动等，未隔离唯一原因。

cached raw对同key预算hold四组均值PSNR高1.607–1.808dB且时间比.480–.723；这说明读取当前图像优于这个固定保持控制，是应保留的局部信号，不能替代对原F0及其强post的比较。逐窗正例、SSIM/edge/全部hold仍保留。静态联系图SYN投影的颗粒/细节与原F0明显不同，ND较平滑；视觉印象不作去噪失败的单一因果或视频感知/MOS证据。

STOP固定quarter、5×5、RGB局部仿射、epsilon1e−4、K2、无算子warp、双线性、原display-F0目标/clip配置及其补救网格；不加warp/guide/更高阶/训练/换周期救分。一般keyframe复用、局部仿射/双边网格、guided filtering均已有直接先例，见[先例80](../../5.prior-art-survey/docs/80-keyframe-affine-operator-and-guided-filter-boundaries.md)。这不是HDRnet原方法复现，不否定学习型双边算子或所有计算复用。

后续需要重审结构与数据假设，明确已完成基线何处提供论文机会，而不是将这次零训练失败再接一轮宽泛训练。论文目标仍active且未完成，尚无确立创新/独立最终/手机收益；原论文84值/snapshot003/PDF020、所有旧STOP/最终封存/7号及并行main/其他worktree保持。
