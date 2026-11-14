# R46 stage002｜实际增强输出的固定坐标状态检验

登记：2026-11-12T17:00:47.029426+08:00；作者：Codex  (sqs_A100)。前提：oracle001完整150输出与127不变量通过，主要clean/noisy MSE比.482196/.507411；它不等于真实流或论文方法有效。

## 固定设定与同mask控制

不训练/不调用CNN/不使用GPU，使用既有R44 square_MSE最终双seed raw缓存；十个既有DEV条件（六SYN64帧/四ND16帧），零保留测试或新数据。half-FB仍只在相邻自己的raw估计，两方向从同一对已到达帧取得，严格因果。一次计算相邻back b和forward f，反向一致性规则沿用half_motion_r4；ordinary的b/mask须与原函数完全相同。

只检验K=8、α=.5，不增加alpha/K/插值/流精度网格。.5是在实际视频已确立的强递归控制，不能由toy中.75选择实际最好alpha。每seed三方法ordinary、same_mask、anchor8，共60质量行；raw和五固定alpha参考复用已完成R45。指标沿用完整evaluate的PSNR/SSIM/edge/GT-change时间误差。

原型先使用原生分辨率位移，以隔离粗坐标网格误差，不宣称低成本实现。设q为锚→前帧、c为前帧→锚，更新q_new=q+W(f,q)，c_new=b+W(c,b)。两方向有效性累计相邻FB、坐标界内及插值mask近1（≥1−1e−6）；输出mask为current→anchor有效且采样anchor→current有效。当前输出用W(h,c_new)，锚内新观测用W(y,q_new)；更新/重置同oracle。被拒当前输出回退raw、锚内观测无效保留h，每8帧重置全部坐标及mask。ordinary仅相邻mask；same_mask普通递归使用候选每帧完全相同输出mask，隔离增加拒绝的影响。

候选persistent状态含previous raw3、anchor RGB3、双向位移4个float32通道及两个bool mask，共42HW bytes；原ordinary为24HW bytes，增加75%，不含瞬时流/临时输出。第一理想阶段不需位移图，该阶段更昂贵；先判断质量信号，不能把它包装成零推理开销/原单帧折叠。若无足够收益，不再优化压缩坐标。

## 实现与接受门槛

先零DEV CPU人工fixture：half-FB backward/mask与历史函数一致；恒定known flow组合方向与解析位移一致，zero/整数运动中央输出复现oracleK8；K1对same_mask普通递归逐像素≤1e−6（归一化）；全拒回退、边界及首帧检查。fixture通过才允许正式读取开发缓存。错误保留后可更正实现，不改科学门槛。

最终ordinary重新评价对R45 fixed .5的四指标差≤1e−5，确保原控制未漂移。每seed/domain，anchor8相对ordinary与same_mask均须PSNR≥−.05dB、时间比≤.90，且所有逐条件PSNR≥−.5dB；还须未被同seed/domain五固定alpha点支配。全部通过才另行登记成本/可行压缩坐标；否则冻结本K8双线性坐标设定，不追加K/alpha/插值/有效性阈值/种子追分。不是用增加同mask拒绝来改低时序指标。

每seed固定30行/1344逐帧行，无墙钟截止/中间选择/自动重试。正式结果完成后分文件提交并汇总所有比较，过程/决定/进展单独提交，阶段推送。大数组仍进入zrh自有data，公开相关工作边界见笔记35；没有新颖性、自然夜间、独立最终集或手机硬件结论。
