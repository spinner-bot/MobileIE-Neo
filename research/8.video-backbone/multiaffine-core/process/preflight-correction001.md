# 首次资格的来源记录修复

2026-11-20；Codex  (sqs_A100)。原CPU3387850/start112858931、session65244已退出exit1，runtime正常标completed_normally=False，0.032629秒、teacher/student图与优化更新均0。实际异常为structure()首帧写`dict(frame=t, **origin)`，bootstrap来源自身也有frame键，Python抛TypeError。该位置之前执行过第0帧FP64 stream和diagonal参考，不能把0学习图误写成完全没有模型数值调用；没有完成结构检查或读取MixedPool训练批。

原config/failure/runtime和19源冻结001全部原样保留并分别提交，不把错误当方法负结果或资格通过。修复副本qualify_attempt002.py只将记录键改为index，并使用独立cpu-preflight002目录/源冻结002。core.py、协议、所有数值/阈值/随机种子/学习预算不变。第二次全资格从新模型开始；原第一次没有优化，不构成继承或追加正式学习。不是因等待超时重启，不自动循环重试至通过。
