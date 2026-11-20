# V1执行入口与接续边界

2026-11-20；Codex  (sqs_A100)。完整算法预算以protocol-v1为准；此页解释自动接续，不扩实验。

原唯一协调器通过project-python、CPU空CUDA mask及CuBLAS确定性环境启动 `run_v1.py --chain 1`，独立session，stdout位于自有data的 `.cache/experiments/topic8-v1/coordinator001-stdout.txt`。原PID2988618/start_ticks109207006，启动来源见process/v1/chain001-launch.json。**不要再次执行该命令来“确保运行”**；先查原PID/start、chain001/state.json、guard原记录、各阶段runtime及真实子进程。状态文件只是索引，不是存活证明；观察超时不表示进程退出。

协调器本身只用CPU，不加载模型。每轮真实11样本先筛0–3组；无合格卡时保留全部原始采样于自有cache、在resource-history.jsonl写简要统计，30秒后再新采样，无墙钟截止。有合格卡后，仍通过既有prebind入口做独立新采样与进程初始UUID绑定，科学worker完整CPU准备后再准入/锁/实际PID-UUID核查。最多一个本链GPU worker，不改既有阈值、他人任务或全局配置。

执行顺序是CPU已通过资格→首次CUDA八丢弃→seed0完整2K＋3×4K→CPU seed0全质量→seed1完整同预算→CPU seed1质量→220官方JOD→完整280行汇总→四固定图。seed0 CPU质量可与seed1的GPU工作并行；两质量正常后才JOD。正式训练只认唯一最终，各1000备份不作DEV选择。没有自动成本扩展或打开封存最终。

prebind在child前拒绝75只等待，不产生科学更新。正式worker资源75只有原保存的完成步边界可续，包含模型、Adam、数据步数及计数；预准备拒绝且没有更新才可零步新进程。JOD资源续接拷贝已保存行并跳过，不重复评分。非资源退出、缺失原终态、已有不完整输出目录都停止供原记录审阅，不能借协调器进行科学自动重试或悄悄覆盖失败。CUDA资格已创建而失败时也不会自动丢弃后重跑。

协调器遇一阶段失败不会终止已经合法启动的其他CPU质量子进程；下一次接续必须逐一核它们，不能仅根据协调runtime失败重做全部链。chain完成只说明上述注册DEV链完成，四图仍需根agent实际查看与独立review记录，方法解释/成本资格/外部与实际手机/创新/独立最终仍待完成。

run_v1自身代码/配置/launch身份单独版本管理；它不改变34个冻结训练/质量源和5个辅助源。活跃源码保持不变，实时state/log先留自有runtime，检查点才保存已核实际前缀，正常终态后分别提交完整结果/解释/过程/进展报告，并按main/no-ff/指定作者作阶段推送。所有旧STOP与其他worktree边界保持。
