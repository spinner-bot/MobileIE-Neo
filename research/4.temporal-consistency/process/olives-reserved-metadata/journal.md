# 既有保留OLiVES来源的纯元数据资格过程

2026-11-13；Codex  (sqs_A100)。上一回合为verified wait；本回合推进独立最终资料准备度，不增加模型/参数网格或解除保留图像。

1. /data挂载/原四句柄及仓库最新0121核验正常。回读R4b作者清单、可见目录与R9协议，确认40 GT不是40作者test，而是31 train+9 test；旧6个作者test已探索，R9另保留video43/71/83，不能把它们自动当新DEV。R9原四dev在作者train中，与正式12训练分开。
2. 原fetch脚本只读一层limit1000、没有分页跟随；本阶段检查HTTP Link，不能假定所有旧计数必然完整。另核R3公开两个样例64文件及R4/R4b/R9实际manifest，保留源角色、排除已取得文件，不读取其图像。标准HF连接器只有repo详情/搜索，没有固定revision逐目录/LFS接口，使用普通匿名官方API，不查登录账户/令牌或他人资产。
3. 先独立提交纯metadata方案，再提交qualify_reserved_olives_metadata.py。命令`CUDA_VISIBLE_DEVICES='' ./scripts/project-python research/4.temporal-consistency/scripts/qualify_reserved_olives_metadata.py > research/4.temporal-consistency/process/olives-reserved-metadata001.log 2>&1`会话53779正常退出0，23:10:18终态、12.301649秒。仅8串行GET，每响应≤2MiB、0重试，完整实际累计383301字节JSON，所有Link为空；目录集合与旧清单相同。
4. 三保留段两侧PNG分别200/225/161，全部同名共同覆盖586；全已列PNG LFS SHA对736个唯一既有资产摘要精确重合0。736而非768是已有R3 example1与R4b video21的32文件重复，旧R4b integrity早已记录，R9与它们均0交集；本次没有发现或伪造新数据泄漏。保留清单首16指针不意味着正式评价仅16帧，未设最终clip长度/显示条件。
5. 原始JSON仅自有data/cache/datasets/OLiVES-reserved-metadata，repo保存config/roles/responses/candidates/summary/runtime，完整stdout另交。0PNG/resolve/viewer/缩略图请求，0解码/flow/缓存视频/CNN/SGD/GPU；缓存保留图像目录仍不存在。没有通过下载图片确认尺寸/捕获时间、观感或语义。
6. followup001核17训练源/协议及27/37评价源保持、原两训练及两协调同身份活跃，最近两seed9700/64000，无终态/新DEV/JOD/重启。data README记录纯元数据覆盖、不证明物理场景独立/预训练未见/可下载，不宣称新标准基准或自然夜景；原R9保留和DAVIS test-dev30 JPEG保持未读。原R75/R76/R77过门槛也不自动解封，最终方法/权重/预处理/评价及读取范围仍须另冻结。
7. 过程/报告0122/入口分别小步提交，阶段fetch仅origin main、必要时显式no-ff合并，不rebase/squash/强推。paper018和全部旧STOP/创新、自然动态、手机及独立最终缺口保持，目标active；原固定训练/评价继续。
