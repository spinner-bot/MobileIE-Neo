# SMOID demo来源资格过程

2026-11-13；Codex  (sqs_A100)。上一回合为verified wait，本回合有限推进外部来源资格，不扩模型/参数或指标网格。

1. 核/data挂载及原R75两训练/原协调/R76补充协调实际PID存活。现有指标协议和paper limitations已联合约束Etemp、质量、JOD及图审，本回合没有另开重复指标诊断。
2. 两次独立GitHub连接器调用均在传输层失败，转普通作者公开raw与GitHub API读取；没有据传输失败判断仓库不存在。API固定当前master为53e81c4，取得树中六份源blob身份；原论文PDF工具403，本回合未重试或宣称新增全文阅读。既有note29论文接口判断保留。
3. 实际读customized_list，三行input和GT路径均相同；test只加载input，不能把demo第二列当独立GT。test还在整个process_video片段上作HE、32帧网络窗口，config MAX_FRAME800；train接口是四通道packed RAW与三通道两倍大小目标。仅代码事实及其有限时间依赖推论，不宣称论文数据没有GT或整个方法非法。
4. 在数据主机请求之前单独提交有限方案，再单独提交资格脚本。源码六份各≤64KiB且Git blob严格一致后才做Cam1固定ID一次匿名HEAD，不执行作者代码/安装旧依赖，不取数组/模型/视频。实际命令为`CUDA_VISIBLE_DEVICES='' ./scripts/project-python research/4.temporal-consistency/scripts/qualify_smoid_demo_source.py`，会话84173正常退出0，runtime11.583981秒。
5. HEAD实际404/text-html/content-length0；本次元数据资格正常结束不等于归档取得成功。0body/解码/CNN/SGD/GPU，按方案停止，不请求另两包、不重试/换来源。源码缓存全部自有data/SMOID，仓库仅记录源身份、状态与解释；config/interface/sources/head/runtime独立于阅读结论提交。404不证明永久删除或其他入口不存在。
6. note60单独保留当前不能增加配对外部验证的结论，仍需真实解决自然动态/手机/独立最终缺口。训练followup001核原17训练源和27/37评价源保持、四原PID/start同身份活跃，无新训练/DEV或重启。过程、报告、接续索引分别小步提交，阶段fetch main后按no-ff规范处理分歧并推送。
