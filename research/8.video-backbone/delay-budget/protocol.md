# D1：一帧等待的信息预算筛查

登记：2026-11-21；Codex  (sqs_A100)。先于D1首次数组读取/flow/指标。接[0202归并](../../../docs/进展报告/0202-T-261121-同输入证据归并与时间信息预算问题.md)及[元数据核查](../evidence-reconciliation/process/delay-source-audit001.json)。前回合为progress：完整896配对/全场景敏感性/先例核查及推送d92527ef3。本次不是A1/L1或R36的参数补救，不解封旧STOP；先查不同的未来信息预算，不把经典均值/flow/一帧等待当创新。

## 唯一固定函数

原R44 F0已clip FP32 RGB[0,255]作为观察X。对同一中心t分别用原half_forward_backward(邻居X,当前X)估计current→邻居的场及FB/bounds；用原align_history采样邻居，无效处回退当前X。过去一帧P1、过去两帧P2、未来一帧N1分别独立估场，无frame/flow合成、GT部署信息或共享未来mask。只共享三个新路线都合法的P1计算，不将N1场/拒绝结果传入因果控制。

past3=(X+P1+P2)/3；center3=(X+P1+N1)/3；center_self=(X+P1+X)/3。固定FP32顺序，三图相加再除3，不额外clip。center3在原t+1可用后才输出t，**一登记帧等待/一未来**；past3与center_self为零未来。past3/center3各中心两个双向场，不是实测同成本，center_self只一个。缓存复用不等于实际新CNN/手机速度。部署F0摊销仍每输入一完整前向，新增flow/warp/缓冲均有成本。

原F0、原自身FB recursive.5、原自身native DIS recursive.5三完整缓存为强参照；不重新生成或更换它们，原rec均从序列首帧递归得到。所有六输出仅取共同内部中心t=2..T−2，即Python [2:T−1]。GT评价相邻t−1→t用原reference_flow[t−1]/mask，首次内部帧没有时间指标；对应GT/输出相同切片，8px边界与原evaluate/PSNR120dB上限保持。没有空域/困难源省略。内部片段不是完整流式视频首尾质量，也不擅称33ms/连续传感器帧。

## 全部来源和预算

仅原反复使用DEV的六SYN条件（bear/lucia/walking各photon20/80，64帧480×854）和四ND（video15/47/91/121，16帧288×512），两原F0模型seed。六条件只有三个合成源，不当六独立场景。原R73引用manifest和R86完成标记及冻结source-audit确定来源/顺序/shape/checkpoint；首次加载必须核自有data路径及payload SHA，不能只核metadata。仅npz访问gt/reference_flow/reference_mask；low只用于最终固定静图，禁止读/运行旧teacher/失败候选。

全部20条件-seed×6=120序列输出、5016逐帧、24域seed均值；每seed418内部帧，两seed836。三路线共享P1，实际新部署双向对2508（P1/P2/N1各836），即5016 Farneback calc；三个路线各自逻辑数为past3 1672、center3 1672、center_self836，不将共享实验数当独立部署总数。GT评价场全取既有缓存，不重估，0新增GT flow/CNN/训练/GPU。新数组仅三路线FP32 npy约11.4GB，放zrh/data项目cache；原三参考只引用，不复制巨量旧资产。每条新npydigest/来源和索引入manifest。

CPU2/BLAS2/OpenCV1/nice5，CUDA_VISIBLE_DEVICES空；依赖只自有project-python。先固定人工128² RGB常量和seed8300随机图三观察相同、零场/全拒回退，各输出对当前差≤1e−6（先归一[0,1]）；provider记录past3访问[t,t−1,t−2]、center3[t,t−1,t+1]、center_self[t,t−1]，未来不进入另两route。固定人工64×80平滑纹理、已知正负flow，核align_history与同一warp/bounds的直接重采样差0、全拒回退精确。资格只核实现，不证明真实运动或PSNR；不另调容差/纹理/窗口救分。

## 唯一继续门与止损

完整120行后才汇总，按SYN/ND×两seed分别，center3相对past3和center_self各要求mean PSNR差≥−.05dB、该域最差完整序列差≥−.50dB、Etemp均值比≤.95；相对原FB rec和原DIS rec各相同质量保护、Etemp比≤1.00。共16比较必须全过，只允许另登记真实整链成本/同延迟结构研究，不自动启动训练/JOD/外部或论文主方法；等帧数/flow数不当速度证据。

任一失败STOP本固定half-FB/三观察/1/3/P1-P2-N1/接受域/内部片段配置，不追加alpha/邻居数/间隔/阈值/flow/teacher/滤波器/训练补救；不泛称所有未来、窗口或视频架构无效。保留所有24均值/16门/80配对序列，以及ND/SYN所有局部正例。绘制完整质量—时间、四参照全配对图、完整16门和固定联系图：每seed首SYN bear_photons20的t32、首ND video15的t8，GT/low/六输出，统一全FOV缩图及中心128²细节，不按结果选图。PNG实际逐张审阅，独立PDF导出；静图不是闪烁/MOS/手机或因果证明。

代码/协议/具名冻结、原完整结果、分析/实际图审/决定、过程/报告分别密集提交并阶段推送research/O1，指定作者/no-ff保持；原84值/snapshot003/PDF020、独立最终、7号、并行main/其它worktree保持，论文目标active未完成。
