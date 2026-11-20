# V1执行入口与接续边界

2026-11-20；Codex  (sqs_A100)。完整算法预算以protocol-v1为准；此页解释自动接续，不扩实验。

原唯一协调器通过project-python、CPU空CUDA mask及CuBLAS确定性环境启动 `run_v1.py --chain 1`，独立session，stdout位于自有data的 `.cache/experiments/topic8-v1/coordinator001-stdout.txt`。原PID2988618/start_ticks109207006，启动来源见process/v1/chain001-launch.json。**不要再次执行该命令来“确保运行”**；先查原PID/start、chain001/state.json、guard原记录、各阶段runtime及真实子进程。状态文件只是索引，不是存活证明；观察超时不表示进程退出。

协调器本身只用CPU，不加载模型。每轮真实11样本先筛0–3组；无合格卡时保留全部原始采样于自有cache、在resource-history.jsonl写简要统计，30秒后再新采样，无墙钟截止。有合格卡后，仍通过既有prebind入口做独立新采样与进程初始UUID绑定，科学worker完整CPU准备后再准入/锁/实际PID-UUID核查。最多一个本链GPU worker，不改既有阈值、他人任务或全局配置。

执行顺序是CPU已通过资格→首次CUDA八丢弃→seed0完整2K＋3×4K→CPU seed0全质量→seed1完整同预算→CPU seed1质量→220官方JOD→完整280行汇总→四固定图。seed0 CPU质量可与seed1的GPU工作并行；两质量正常后才JOD。正式训练只认唯一最终，各1000备份不作DEV选择。没有自动成本扩展或打开封存最终。

prebind在child前拒绝75只等待，不产生科学更新。正式worker资源75只有原保存的完成步边界可续，包含模型、Adam、数据步数及计数；预准备拒绝且没有更新才可零步新进程。JOD资源续接拷贝已保存行并跳过，不重复评分。非资源退出、缺失原终态、已有不完整输出目录都停止供原记录审阅，不能借协调器进行科学自动重试或悄悄覆盖失败。CUDA资格已创建而失败时也不会自动丢弃后重跑。

协调器遇一阶段失败不会终止已经合法启动的其他CPU质量子进程；下一次接续必须逐一核它们，不能仅根据协调runtime失败重做全部链。chain完成只说明上述注册DEV链完成，四图仍需根agent实际查看与独立review记录，方法解释/成本资格/外部与实际手机/创新/独立最终仍待完成。

run_v1自身代码/配置/launch身份单独版本管理；它不改变34个冻结训练/质量源和5个辅助源。活跃源码保持不变，实时state/log先留自有runtime，检查点才保存已核实际前缀，正常终态后分别提交完整结果/解释/过程/进展报告，并按main/no-ff/指定作者作阶段推送。所有旧STOP与其他worktree边界保持。

## 已发生的资源接续修正

2026-11-20T13:37:20.170968+08:00；Codex  (sqs_A100)。chain001已于13:32:39终止，旧PID/三个原句柄已核缺失，不再视为活跃。首正式worker在warm241按资源规则保存后返回75；attempt002继承241但内部准入拒绝、CUDA分配0/新更新0，因此runtime没有新checkpoint。原协调器误将非零继承计数视为必须有本轮新checkpoint，断言停止。旧失败与全部原runtime保持，没有科学失败或更新丢失。

仅run_v1的元数据接续逻辑已修正：当实际counts等于config的inherited_counts、尚无GPU/CUDA分配，且能找到计数/seed/路径/摘要全部匹配的原保存marker时，可沿用上游完整状态；任何未保存的新更新或跨seed状态仍拒绝。实际241记录的回归检查正常，0模型执行/SGD/新像素；不修改34训练/质量源、5辅助源、算法或门槛。

当前原chain002于13:35:43启动，PID3003196/start109313161，使用coordinator002-stdout.txt；新配置绑定修正后源码。progress004核同身份live及实际forwarding日志，沿用原241状态等待首次attempt003，不重跑资格或从0训练。下一以chain002及其原worker/guard/runtime为准；本页前述chain001启动段是保留历史。

## CPU接续已启动（后续实际状态）

2026-11-20T14:20:00.852101+08:00；Codex  (sqs_A100)。共享GPU多次拒绝/资源保存后，新增[CPU后端修订](cpu-backend-amendment001.md)并独立冻结4个执行源。原chain002只在所有数值worker正常保存结束、coor无child时以Python SIGINT退出空闲watcher，原False/KeyboardInterrupt及全部GPU历史保持；未向任何数值worker发信号，不再监视旧PID作为活跃。

当前原cpu-chain001协调3024380/start109481893、worker3024402/start109481902，两线程/nice5、空CUDA mask，真实从GPU358的完整状态继续359。warm2000已正常，三控制GT适配继续。新CPU控制器完成两seed原预算及CPU质量后才启动原post chain003评分/汇总/联系图，训练期间不要手工启动chain003或旧GPU链。当前以process/v1/cpu-chain001/state.json、原PID/start、attempt006真实trace/marker/runtime和progress009为准。原34＋5源及新增CPU4源活跃期间保持冻结；后续过程写本页外的journal/报告，避免修改已冻结执行修订。

## 2026-11-20T15:03:42.532564+08:00：首seed已正常，当前改核新的原句柄

原attempt006 worker3024402/start109481902于15:00:21正常完成warm2000＋三组各4000、14000总更新，已核/proc缺失，勿继续将其视为live或重做该seed。原CPU协调3024380/start109481893仍运行；实际自动启动质量seed0=3057709/start109820957和独立训练seed1=3057710/start109820958，两者各2线程/nice5/空CUDA mask。先核这三个/proc身份与各自原日志/runtime；progress012是当时核验记录，不是实时状态。两seed/质量完整后仍由同协调器调用原post chain003，当前不要手工启动；全部官方JOD、共同表、四图和根agent实际审阅尚待。报告0173记录本阶段，原34＋5＋4源与全部门槛保持。

## 2026-11-20T16:01:07.457759+08:00：两seed训练均已结束，当前只核最终评价句柄

seed1于15:58:09正常完成原14000，与seed0合28000；旧训练3057710已退出，勿重启。当前原CPU协调3024380/start109481893和新质量seed1=3095929/start110167820，15:58:10实际启动；先核这两个/proc/start与原runtime。两CPU质量正常后由原协调器自动调用post chain003，当前仍勿手动启动重复链。完整官方JOD/共同表/四联系图后仍需根agent实际阅图和解释。用户要求按近期速度估算sleep，参考同完整seed0质量905.255895秒、扣除已过时间并覆盖大部分余量，醒后再核同句柄；不以估算到期重启或改变原预算。详报告0174/progress017。
