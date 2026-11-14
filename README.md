# MobileIE-Neo

面向手机端的轻量图像增强模型研究：在 MobileIE（极轻量实时图像增强 ConvNet）基础上，解决其应用于视频时产生的帧间闪烁问题。

## 当前研究状态

截至2026-11-14，项目仍处于方法探索阶段，尚未建立可投稿的新方法。接续先读[进展报告0133](docs/进展报告/0133-T-26111409-正式训练双种子两万步里程碑归档.md)、[4号研究入口](research/4.temporal-consistency/README.md)和[服务器约定](AGENTS.md)。主线在 `main` 推进，移动演示由独立支线处理。

[原保留实拍来源](research/4.temporal-consistency/data/external-olives-reserved/README.md)只做纯目录元数据资格，三段586对帧名、0像素/模型/解码；不增加已完成独立基准数，不开放为DEV，也不因原门槛通过自动解封。最终方法、权重、预处理和完整评价范围须另冻结；原训练与既定完整评价链继续，paper018和研究缺口保持。

最近[progress015](research/4.temporal-consistency/process/exploration-r75/progress015.json)归档R75两seed各20000/64000、各20000可恢复marker和完整行日志前缀；原训练及原/补充协调器继续，17/27/37源保持，无中途DEV/JOD或新方法结论。R77只预先登记同架构独立空间控制并通过CPU人工输入资格；必须先过原八门槛、self4四支持及两套实际图审才投入额外训练，否则停止该投资。当前没有新正式训练/GPU，未完成完整trainer/evaluator，不把current/self4的输入干预当独立空间训练或创新。SMOID demo来源核查也未取得新配对外部基准；paper018及验证缺口保持。

先前[progress003](research/4.temporal-consistency/process/exploration-r75/progress003.json)归档R75两seed 8100/8000步和各8000检查点；后续以上述progress015/实际runtime为准。提前登记的[R76固定同帧self4控制](research/4.temporal-consistency/docs/exploration-r76-self-conditioning-control-protocol.md)已补齐质量/感知/100条件汇总/20固定图入口，编译与metadata屏障资格通过、37源冻结；补充协调39049只等待原完整结果，0新增GPU/训练或真实DEV读出。四倍计算的解释控制不改原八门槛、不当新方法；真实数值和实际图审仍待。R-DID获取未完成，不能增加已验证基准数。

R75公开因果参考的正式训练已启动：发布ReBotNet-XS、自启动native384四帧、两seed各64000，实际在同组0/1号卡按余量共享。首次启动快照为各300步，后续状态以上述progress015及真实runtime为准；完整最终质量/感知尚未执行，不能从训练loss宣称方法成功。最终评价协调器会核真实训练句柄、等两final后接续固定流程；训练/评价源码已冻结，见[首次启动快照](research/4.temporal-consistency/process/exploration-r75/progress001.json)及[执行边界](research/4.temporal-consistency/docs/exploration-r75-evaluation-execution.md)。共同表的ND输入为288×512原生裁块，非整幅1080p；普通参考适配不是创新或原500K充分复现，旧STOP保持。

定向来源核查已确认ReBotNet作者发布架构，CPU人工输入资格通过，原代码/许可已固定入库。固定token/GT首帧/500K原训练与任意原生分辨率的差异见[57号阅读](research/5.prior-art-survey/docs/57-causal-reference-code-feasibility.md)；代码资格阶段没有训练/质量/视频像素/GPU结果，不称方法或原论文充分复现。

前阶段另完成原生分块CPU检查和两次丢弃共享GPU更新，原试跑进程已退出释放、状态不继承；其[正式预算](research/4.temporal-consistency/docs/exploration-r75-causal-reference-protocol.md)现已由上述正式作业执行。资源资格、活跃训练与最终有质量收益的方法分别记录。

R74冻结空间映射/RGB传输的[固定检验](research/4.temporal-consistency/docs/exploration-r74-motion-supervision-protocol.md)已完整结束：16000 motion更新、80新CPU质量、80感知、20固定图实际审阅及120条件联合表全部保留。16门槛全部失败，按[完整决定](research/4.temporal-consistency/docs/exploration-r74-decision.md)停止其局部搜索，未运行成本实验；当前无活跃R74任务，不依据历史提示重启。共享GPU实际设备问题已透明纠正，后续[入口](research/4.temporal-consistency/docs/prebound-shared-gpu-entry.md)在整个子进程前绑定并核实际UUID，非0卡零评分检查通过；旧错误记录和所有原分数保持。创新和适当因果视频参考仍需继续建立，不把普通模块组合或负结果称方法成功。

R73原生一/二阶历史矩进入第一次FST的固定检验已全部正常结束：24000正式更新、六最终导出、200新CPU质量、240感知、280联合条件和20固定图实际检查完整保留。20项联合门槛全部失败，按[完整决定](research/4.temporal-consistency/docs/exploration-r73-decision.md)停止该组合的局部网格；没有因此建立新方法。当前无活跃R73任务，不依据历史报告中的活跃提示重启旧训练或评价。

R28/R34/R37/R44/R45/R48/R64等既有具体候选已按各自完整结果止损，不重开相似超参数搜索。R70传统视频滤波、R72公开REDS开发源强参考均已正常收口；这些开发证据不等于独立最终测试或自然夜间验证。CRVD仅一个训练时刻四份原生RAW文件的有限输入资格已完成，未进行模型评价，不计作完整视频基准。

现有[论文工作稿](paper/manuscript.pdf)为已审阅build018、22页/14表，包含完整R72原生REDS合成开发对照及R73原生状态主候选/全部注册参考的完整失败证据；尚缺可辩护创新、画质—时序—效率联合改善、自然动态配对、实际手机成本及独立最终验证。轻量移动端目标需要完整成本证据；原MobileIE的4075参数不是用户给定的硬性参数上限，含历史状态的模型也不能称零推理开销。

2026-11-12 03:06，自有环境和数据迁移已核验完成，两个仓库相对链接已入Git，后续统一用 `./scripts/project-python`；不重复迁移。详情见[迁移日志](research/4.temporal-consistency/process/asset-migration/journal.md)。代码、过程、结果、报告分别小步提交，阶段SSH推送，保持 `--no-ff`。新GPU任务须先检查共享卡的显存、利用率及计算进程，确认双方资源余量后可同卡并行，使用同组最多4张卡。

## 目录结构

| 目录 | 用途 |
| --- | --- |
| `MobileIE/` | 上游官方代码（vendored，提交 `7030d12`）与预训练权重，各工作共用 |
| `docs/` | 进展报告（Markdown） |
| `specs/` | 项目规范与需求文档，如 `项目规范/版本管理规范.md` |
| `research/` | 各阶段工作的中间资料与辅助内容（前期准备、复现、视频闪烁） |
| `assets/` | 图片等静态资源 |

## 项目规范

版本管理遵循 [`specs/项目规范/版本管理规范.md`](specs/项目规范/版本管理规范.md)：频繁提交、SSH 提交、AI 提交作者标注 `claude code (local)` / `Codex (local)`、禁用 fast-forward 合并。本服务器主线按用户约定使用作者 `Codex  (sqs_A100)`，覆盖旧本地署名默认。

## 快速开始

```bash
git clone git@github.com:spinner-bot/MobileIE-Neo.git
cd MobileIE-Neo
```

服务器运行资产固定在 `/data/Users/zrh/data/MobileIE-Neo`；仓库 `.venv`、`.cache` 已分别指向该目录的独立环境和缓存，并将相对链接入Git。运行统一用 `./scripts/project-python`，例如 `./scripts/project-python -c 'import torch; print(torch.__version__)'`。迁移过程及恢复记录见[迁移日志](research/4.temporal-consistency/process/asset-migration/journal.md)；禁止借用其他用户环境或在/root安装项目依赖，详见[路径与环境规范](specs/项目规范/路径与环境规范.md)。

历史进展及失败过程完整保存在[进展报告目录](docs/进展报告/)和Git历史中；当前状态以上述接续入口及实际终态为准。
