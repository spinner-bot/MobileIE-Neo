# R74完整感知评价的共享GPU执行

2026-11-13；Codex  (sqs_A100)。仅执行资源修订，落实用户最新同卡共享授权。原CPU训练、质量和固定图不改、不重做。42冻结文件保持原字节。

先前R66 reference002、R69、R70、R73完整感知运行的原runtime均正常，同10开发条件、原尺寸、ColorVideoVDP0.5.7/standard_fhd/30fps/replicate/gpu_mem=2，CUDA峰值allocated360763392字节（344.05078125MiB）。这不是完整进程显存；上下文、缓存、库分配另计。REDS原720p的429958656字节不能替代本轮10条件的配置核对。此次仍保守估计进程增量4GiB，缓存分配器限额2GiB，至少为已有任务留8GiB增长和1GiB余量。原包的gpu_mem=2不修改。

新根scripts下wrapper只替换原evaluator的cuda_headroom调用和predict调用外的资源调度；原evaluator.main、官方metric构造、人工CUDA资格、predict数值操作、全部80条件/3584帧、缓存、统计、名义播放速度及16门槛保持。每个完整序列同步计时后主动暂停，初始计算占空比目标25%。暂停在predict返回之后，不插入视频帧/改变帧率或分数。每序列检查显存；余量消失或连续三个序列繁忙采样≥90%就停止本项目评价并保留已产生的部分结果，禁止自动重做/混合选分。

只能在两seed完整正常、两CPU评价与summary完整正常、20图renderer完整正常之后接续。若原finalize002已开始感知子任务，继续原任务，本wrapper不得并行或重复评价。若它仍仅在旧空卡资源等待，可在核PID/start/cmd且无子任务后，仅终止该控制等待；保留原False runtime/KeyboardInterrupt/资源记录与明确handoff原因。没有这样的终态与handoff记录，新wrapper拒绝执行。不会为此中断正式训练、渲染或任何科学子任务。

wrapper在原evaluator完成真实CPU输入及CPU metric准备后才连续采样11次/间隔2秒，选一张有余量的卡并持有同项目锁；再次核余量后创建CUDA上下文。原输出目录必须尚不存在，任何已开始的科学评价失败不得自动重试。wrapper源、准入、指标执行、runtime及后续summary分别记录。资源不足且尚无CUDA/评分时可另编号重新准入。

原协调的中断不代表方法失败，新评价失败不代表算法已过门槛；完整感知后继续原summary --perceptual及20图实际审阅。只有全部16门槛通过才运行原CPU成本实验；否则按原协议止损。资源调度及等待时间不能当模型推理/手机延迟，未测对方任务吞吐不能声称零减速。
