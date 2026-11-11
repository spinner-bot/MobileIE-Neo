# R20 动态聚合顺序（进行中）

创建：2026-11-11T21:48:32.613420+08:00；作者：Codex  (sqs_A100)；原因：把R19可用信息结论放回运动/事件条件，避免静态排序误推到视频。

固定R14 native plain双seed4000、46旧条件、八方法/两reset共1472逻辑行；alpha.1/.5全报，不新训练。共享teacher半尺度FB，pre/输出warp-post/先对齐输入后增强-post与原recursive half01；零GT motion部署。三个CPU fixture通过，核验仿射交换、平方Jensen差及有限reset/同帧精确行为。

raw/recursive逐帧逐片回放R14≤1e-10门槛，alias只在raw或无事件精确条件复用，预计1072真实指标/400alias；神经阶段不当全pipeline成本。源码/检查/协议已分别提交；计划空闲GPU2/3分别跑seed0/1，两次预检后执行，无墙钟截止。阶段完结按spinner-bot专属SSH推送。
