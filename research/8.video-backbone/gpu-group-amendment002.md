# 官方评分组执行修订002：同步外层和内层组限定

2026-11-20T16:27:05.264825+08:00；Codex  (sqs_A100)。触发：001路由外层选择6号卡后，冻结runtime_io所引用R79 acquire_gpu仍硬编码0–3，首attempt001正常资源退出75、JOD0/比较帧0/CUDA分配0；attempt002由原资源循环启动，须等待其原资源终态，不中止数值child。此为漏查内层组限制的操作缺口，不是模型/评分科学失败。001原源码、配置、准入、原runtime和错误原因完整保留，不回写成功。

新增shared_group_v1.py逐字抽取R79 acquire_gpu，只改允许卡组为4–7及报错文字；ROOT/write来源是相同项目入口。连续11×2秒/余量/增长保留/锁/即刻复查/2GiB allocator/预初始化绑定/实际PID UUID验证/SharedTrainingSchedule完全相同，后两者已读源码无额外0–3限定。没有放宽任何内存或利用率条件，不触及其他用户进程。

新增perceptual_group_v1.py从原perceptual_v1.py复制，只将acquire_gpu导入换成上述执行助手，并核/记录独立执行源冻结；220预测/9856比较帧、官方包/显示/fps/precision、predict调用/序列循环/资源checkpoint/统计和错误行为逐行保持。原34科学质量/5辅助/CPU4及001执行源都不修改。新run_post_group_v2.py仅将001 route freeze改为002、调用上述指标执行副本；两seed训练/CPU质量完整前置核验和既有完整汇总/四图不变，不重训/重评普通质量。

旧chain004仍由未改001代码运行。仅在其资源间隙、实际无child时正常退出元数据watcher；如果真实child尚活跃即恢复watcher，等原资源终态后再尝试，不向任何数值worker发信号。保留全部旧零评分资源尝试、原coor False/KeyboardInterrupt及完整日志。新chain005在旧watcher实际终态后以同组4–7新采样/预绑定启动；旧尝试0行不会重算已得分数，后续非零资源prefix照原resume规则保留。首次新GPU实际PID/UUID匹配和真实评分行是执行通过证据，外层选择本身不能替代。

本修订仅修复物理GPU执行范围，全部指标公式、数据、模型、原预算/止损、最终封存和论文状态保持；不将操作路由成功或部分JOD当方法有效。长等待继续按实测速率sleep，减少反复激活。
