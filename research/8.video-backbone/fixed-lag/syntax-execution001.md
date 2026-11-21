# F1首次执行的纯语法修订

2026-11-21；Codex  (sqs_A100)。原3936682/start身份已在launch001记录；实际PID已无，原stderr在[失败记录](process/launch-failure001.json)。evaluate.py在解析阶段因config字典最后一个布尔字段漏写=True产生SyntaxError，未执行模块/资格/像素/flow，results目录不存在。这是可避免的准备错误，不能当方法失败或观察超时重启。

原19冻结文件/协议/evaluate.py/launch.py不改。evaluate_syntax001.py逐字派生：只补上述配置字段的=True，另增执行源验证函数、入口/末尾验证及config执行冻结字段；所有数值函数/循环/数据/门槛/预算原样。launch_syntax001.py只换新runner/launch002/worker002并验证新增执行源。两新文件及core/summarize/f1_io已用compile检查语法，无模块执行或像素读取。

新增三执行源（修订说明、派生runner、派生launcher）独立冻结；原无数值作业/权重/结果可继承。仅在原句柄已终态后启动一次修正执行，原失败不覆盖、不把launcher返回当数值完成，不改变任何旧STOP或科学条件。
