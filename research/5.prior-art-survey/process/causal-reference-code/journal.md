# 因果参考来源与功能资格过程

2026-11-13；Codex  (sqs_A100)。仅本阶段，后续以实际runtime和新协议为准。

1. 确认/data挂载、main干净/0064744a、作者/no-ff配置、自有环境与GPU共享规范。R74所有科学任务已经结束，不重启。旧远期目标仍active。
2. 定向核EVRNet作者v1/主页及一个第三方SR文件。未确认官方代码，第三方通道/激活/scale不符；不导入、不扩大同名检索。来源与访问限制独立source-reading001保存。WACV HTML本次403，旧PDF不重试。
3. ReBotNet项目Code交叉认证作者仓库，固定dff81d3完整树；按许可/原架构/原配置与来源/可选依赖分别提交。复制时许可和无末换行配置各多一个换行，随即独立纠正；架构原字节不改。未下载第三方数据、权重或加载器。
4. 自有project-python安装匹配CUDA的vision0.24.1+cu126（官方cu126、no-deps），再安装固定timm/hub/safetensors及所需依赖（PyPI）；均正常结束，各完整流分别提交。完整pip报告在自有cache，精简版本/公开来源另存environment-additions001。Torch2.9.1+cu126保持。两次临时版本字典查询分别因漏import和缺hub发行元数据失败，未启动模型或改变数据，随后真实CPU资格给出完整版本与真实路径；无借用/全局安装。
5. 先提交CPU功能脚本，再以CUDA_VISIBLE_DEVICES为空运行一次：`./scripts/project-python scripts/qualify_rebotnet_cpu.py --out research/5.prior-art-survey/process/causal-reference-code/cpu-qualification001.json`。真实PID4151838正常退出，3.361秒；无SGD/质量行/视频像素/CUDA初始化。有效384及同token矩形、历史梯度、全部预定失败形状与部分MAC完整保存；原结果与原终端流分开提交，没有为得到合格结果重跑。
6. 57号阅读追加实际结果/500K原训练边界，0113进展报告单独提交，接续入口按文件更新。论文不添加功能数字当质量结果、不重建PDF；没有正式新训练登记。下一才是有界自启动分块与共享GPU需求实测协议，不把发布架构适配当创新。
7. 阶段推送前只fetch origin main、检查分歧/diff（排除原终端log空白）、按需显式no-ff合并。保持小步历史，不rebase/squash/强推。实际推送结果以终端及后续阶段记录为准。

本阶段推送：origin main相对本地落后20/领先0，不需要merge；SSH推送9325e9c9成功。首次diff空白检查报告原样vendor架构/配置的上游尾空白，保留原代码，不为格式改第三方计算来源；重新排除这两个明确原样文件和原终端log后自有改动空白检查通过，工作树干净。后续GPU小试另立协议，不将此推送误作训练已启动。原人工书写13:15非自动事件时间，已移除；实际先后以Git与运行at/ended记录为准。

## 原生分块与共享资源阶段

1. 先独立提交rebotnet-reference-preflight协议，再提交原架构外部tiler、自启动预测和CPU解析资格脚本。384 tile/256步幅/末尾完整覆盖/正三角归一/短边replicate；原架构不改。CPU命令`CUDA_VISIBLE_DEVICES='' ./scripts/project-python scripts/qualify_rebotnet_native_cpu.py --out research/5.prior-art-survey/process/causal-reference-code/native-qualification001.json`一次正常，PID4154286/13:12:29结束/2.140552秒，0SGD/CUDA/视频像素。原结果和终端流分别提交；没有为通过资格重复运行。
2. 单独提交train-only384资源重裁准备及两次丢弃更新脚本。源来自已验证MixedPool首窗口，非新的均匀384正式采样；不读DEV/test，初始所有资料在CPU准备。命令为`./scripts/project-python scripts/prebind_shared_gpu.py --record research/5.prior-art-survey/process/causal-reference-code/shared-trial001-parent.json -- ./scripts/project-python scripts/trial_rebotnet_shared_gpu.py --out research/5.prior-art-survey/process/causal-reference-code/shared-trial001`，终端流存同层shared-trial001.log。
3. 父13:14:50完成原11×2秒采样选择0号，child4155198整进程前UUID绑定。真实CPU资料准备后再11×2秒，实际GPU0匹配，无改其他任务。数据读取期间无GPU分配；没有用过期父采样代替CPU准备后的新检查。
4. 两次丢弃Adam更新/八视频完整四帧反传正常，13:17:08.379 child正常终态、父13:17:09.305返回0。模型/optimizer未保存或继承；正式更新/质量/检查点均0。peak allocated678528000B/reserved710934528B，10条资源观测own最高1242MiB/最少空闲37677MiB；他人观测显存2026MiB不等于测过其吞吐。原FutureWarning保留，不改原始架构导入。结果、配置/准入/实际卡/观测/终态、原终端流及退出核验分开提交；本PID已不存在，GPU上下文已释放，不重跑小试。
5. 资源结果后才独立登记R75正式协议：2seed×64000、原生均匀384四帧BPTT/Charbonnier/cosine；final40新质量/40感知/20固定图，强rec和same-weight current双域双seed完整八门槛。正式初始25%合作占空高于短试跑15%，已明确写入协议，仍逐次新准入和持续监测，不承诺资源永久独占。普通参考适配不是新颖性/原500K充分复现。
6. 0114进展报告单独提交，接续入口按文件分别更新，明确正式数据/训练/评价代码和CPU数据资格**尚未完成，正式GPU未启动**。下一实际实现，不根据本登记自动假设已有checkpoint；不重开任何旧STOP。paper018未重建或加入功能/资源假质量。原先的源码资格状态只适用于当时，最新状态由0114覆盖。
7. 本阶段结束前只fetch origin main、查分歧与自有diff空白并SSH阶段推送；保持指定作者、no-ff及全部小步历史。原样vendor空白和完整终端流按已说明范围保留。远期目标active，不将阶段完成标为论文目标完成。
