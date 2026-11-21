# F1：复用递归状态与正向场的单步创新平滑筛查

2026-11-21；Codex  (sqs_A100)。先于F1首次人工资格/像素/flow。上一回合0204来源时序核查及推送21d625b20为progress。见[先例83](../../5.prior-art-survey/docs/83-fixed-lag-smoothing-and-motion-state-boundaries.md)，普通固定时滞/运动补偿不是创新。F1是独立已知参照：完整原DIS递归过去状态＋一未来创新，不是D1三观察/alpha/flow补救或R36方差估计续跑，不解封任何旧STOP。

## 固定函数与匹配控制

Z_t为原冻结R44 F0 clip后的RGB255，F_t为其原native DIS recursive.5缓存（从首帧开始，不改底座）。在原Z_t/Z_{t+1}上一次原R86同参数DIS正反估计：back=current(t+1)→previous(t)，forward=t→t+1；两向分别按原1+.05范数/界内规则构造FB有效域。back及其mask必须保持原实现值；forward以前仅用于一致性，现在额外返回，不能声称运动估计免费或没有flow。

在t网格用forward采样未来值，W为原cv2双线性/边界实现。M=forward有效域∩采样未来back_mask≥1−1e−6∩采样界内，避免接受不可靠未来状态；无效精确回退F_t。固定两新增路线：

- innovation_lag1：raw S_t=F_t+.5 M·W(Z_{t+1}−F_{t+1})。
- naive_future：raw B_t=F_t+.25 M·(W Z_{t+1}−F_t)。

两者最终clip[0,255]显示/评分，各自未clip范围/越界比例另记。共享相同场/M、未来1及递归F_t；naive是同一标量等效系数的已知直接未来混合，防止把较弱/较强历史权重当机制收益。在零运动且所有接纳的标量.5递归中两者相等；一般图像上因双重重采样/拒绝不必相等，差异不自动证明更优统计。

稳态标量随机游走R=1/Q=.5/P=.5/Ppred=1/K=.5/G=.5/Psmooth=.375只用于解释选定常数，先用Fraction核恒等；不估计真实RGB噪声/协方差、不宣称校准RTS最优或方差上界。反向光流采样不是协方差加权转置。

## 来源、完整范围与计数

原D1/旧R73/R86共同十DEV条件×两F0 seed，六SYN条件3源64帧、四ND源16帧，同t=2..T−2公共内部中心，原GT字段/8边界/PSNR120dB保持。六输出：F0/F0_FB/F0_DIS、原D1 center3、naive_future、innovation_lag1；不重新生成四原参考。读取原source-audit/D1完成标记/manifest，payload首次加载核自有data路径/SHA/shape/finite。GT只用于指标，low只在最终固定静图读取，未来只进入两新路线。

20条件-seed、120序列/5016逐帧/24均值；836公共中心、实际新DIS双向对836/1672 calc，人工资格另计；原GT缓存408唯一评价对复用，0新GT flow/CNN/训练/GPU。实验只在公共中心重估场；未来实际整片部署从首帧运行全部876相邻对，可在原递归同一对DIS估计上复用两向，仍每输入完整F0一次，额外warp/FB mask采样/缓存/等待有成本。缓存速度不是部署速度。

两新增路线40 npy约7.57GB放zrh自有data，原四参考仅引用。CPU2/BLAS2/OpenCV1/nice5，CUDA_VISIBLE_DEVICES空。时间以登记帧为单位，center3/两新路线均一未来；原其他参考零未来。[0204契约](../cadence-audit/results/audit001/time-contract.json)保持，walking/新增源不补fps，非实测33/40ms或手机吞吐。

## 人工资格（先于真实像素）

Fraction核上述标量关系精确成立。固定seed86001/128×160灰度平滑纹理RGB、已知3/−2位移：新双向接口和原FixedDIS的back最大差0、mask完全相同；各一次双向对，共2对/4calc，无额外motion accuracy门。原默认getter全部一致，不换fixture/参数。

固定seed8400/64×80 RGB、已知零场/全接纳，原.5递归值构造F_next，归一化后innovation与naive差≤1e−6；未来残差0和全拒均精确回退F_t。固定正负(±1.25,−.5)场，由同一双线性/有效域构造F_next，检查未clip差满足.25(F_t−W_forward W_back F_t)在共同可靠域归一差≤1e−6；该代数不是真实运动/数据性能。所有函数有限，失败记录，不调常数/容差救分。

## 完整判定与止损

120行全部完成后，按SYN/ND×两seed对innovation_lag1作16必要比较：对naive_future要求Etemp比≤.97；对原F0_DIS≤.95；对原F0_FB和原D1 center3各≤1.00。各比较同时要求均值PSNR差≥−.05dB、域内最差完整序列差≥−.50dB，不省略空域或不利来源。16全过只允许另登记真实整链费用/感知评估；费用需相对原FB≤1.00、相对原DIS≤1.10，完整CPU费用必须真F0/双向DIS/所有mask/warp/布局/状态，不用缓存IO代替。仍不能直接称创新、手机或独立最终通过，不自动开启训练。

任一失败停止本固定native DIS/单步创新/增益.5/直接混合.25/共同可靠域/clip及增益/lag/flow/阈值/噪声先验/训练补救网格；不扩失败成本/JOD/外部/手机，不据此否定全部平滑/未来或骨干。保存全部24均值/16门/80配对与所有局部正例，完整后才判读/渲染。四固定PNG（全点、四参照配对、16门、原bear_photons20 t32/video15 t8×两seed全FOV/中心128联系图）实际审阅，另存PDF；静图非视频MOS或唯一原因。

源/协议/具名冻结、资格/原结果、分析/图审/决定、过程/报告分别密集提交并阶段推送research/O1，指定作者/no-ff/自有data保持；原84值/snapshot003/PDF020、独立最终封存/7号/并行main和其他worktree保持，论文目标active未完成。
