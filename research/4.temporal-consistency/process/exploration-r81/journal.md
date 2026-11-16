# R81过程：一次训练图块噪声节点干预

2026-11-17；Codex  (sqs_A100)。接续R80完整收口，main干净且origin/main一致。上一goal轮为实际进展：完成原160官方JOD、24联合比较和20实际图审并推送55小步；没有活跃旧任务，不重新启动。

先回读原MobileIELLENetS与FSTS代码、R24/R27/R45协议/决定及旧消融方案，发现argmax变更不等于max数值跳变；max在无穷范数下1-Lipschitz，故不直接将通道索引解释为闪烁根因。外部搜索不作为创新证明，CRISP是notes08/13已读先例的回访；RetinexMCNet普通PDF403后停止，旧403记录继续保持，未读全文/绕过，source-reading另交。

协议948da188先于本轮新像素/噪声/模型提交，执行前5c006298明确128唯一噪声实现由两权重共同使用，256是noisy模型调用，不虚增独立样本；其余预算/主门槛不改。三代码分别提交并py_compile，17登记源a5b5bbf0冻结；没有丢弃SGD/优化器或GPU资格。

原CPU934742/start79922028于03:57:12启动，03:57:20正常结束6.274232秒；bmx-bumps和boat两原训练源第0帧中心128、ph20/80、两seed，共8条件/40指标/1280逐draw比较，256 noisy+4 clean core和12原forward，共272骨干/1040tail。源17不变、两权重冻结、12重组函数比较由正常终态前断言核≤2e-5（未保存观测max，不声称精确0）。0DEV/SGD/GPU，原PID已退出，未重跑。

主门槛0/8，local参照PSNR全8改善但boat相邻差只降7.52%–9.19%，bmx ph80反升9.69%/10.69%。global/both参照与当前含噪表征不匹配，global PSNR下降2.00–17.99dB；保留unclipped和clipped方差，不把clip掩盖的振幅当成功。winner差异是相对固定clean参照，不是相邻draw切换率，不能从它认定闪烁因果。

render原执行04:02:16正常1.659408秒，8张固定draw0/1/31图全部已原尺寸view_image实际查看，actual-review001独立保留原renderer false。无模型或SGD重复；完整结果每文件/8图逐张、实际审阅、原stdout、progress001/决定/过程/报告分别提交。运行数组仅自有data/cache，未取新视频/权重或打开最终像素。

按决定降低仅稳定原localgate优先级，不扩max/mean/softmax、温度/滞回/EMA或draw/源/seed网格，不启动正式训练/DEV/GPU/手机，不将静态训练干预写成自然视频主成绩或普遍上界。paper020不扩页，旧STOP与有效方法/联合收益/自然动态/手机/独立最终缺口保持，论文目标active。阶段fetch仅main/SSH推送，无分歧则不合并，有合并明确no-ff。
