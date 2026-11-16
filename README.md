# MobileIE-Neo

面向手机端的轻量图像增强模型研究：在 MobileIE（极轻量实时图像增强 ConvNet）基础上，解决其应用于视频时产生的帧间闪烁问题。

## 当前研究状态

截至2026-11-16，项目仍处于方法探索阶段，尚未建立可投稿的新方法。接续先读[进展报告0151](docs/进展报告/0151-T-26111618-完整因果证据入稿与二十四页审阅.md)、[4号研究入口](research/4.temporal-consistency/README.md)和[服务器约定](AGENTS.md)。主线在 `main` 推进，移动演示由独立支线处理。


完整R74负结果与R75/R76最终证据已接入[论文工作稿](paper/manuscript.pdf)：build020为24页、16张数值表、13项引用，无编译warning/未解析引用，整份缩略和变化页已实际查看。旧14张数值表不变；两张新表保留全部双seed读出与八主门槛/四支持。R75八项0/8、R76支持1/4、R77停止额外投资保持；这仍是内部证据稿，创新方法、自然动态配对、手机和独立最终未完成。下一回到不同信息/学习问题的假设选择，不重开旧网格。

[原保留实拍来源](research/4.temporal-consistency/data/external-olives-reserved/README.md)只做纯目录元数据资格，三段586对帧名、0像素/模型/解码；不增加已完成独立基准数，不开放为DEV，也不因原门槛通过自动解封。最终方法、权重、预处理和完整评价范围须另冻结；原训练与既定完整评价链继续，paper018和研究缺口保持。

最近[progress032](research/4.temporal-consistency/process/exploration-r75/progress032.json)已确认原两seed各64000、共128000更新及R75/R76完整评价均正常结束，旧训练/协调进程已退出。观察间隔38K–64K检查点逐份归档，完整流、数值/感知、两套各20固定图及实际图审均已保存；17/27/37源保持。R75八联合门槛0/8，R76有限支持1/4，按预登记停止R77额外128K训练投资，无新正式任务启动。

完整[因果参考表](research/4.temporal-consistency/report/exploration-r75-joint/tables.md)为80条件/16均值，[self4补充表](research/4.temporal-consistency/report/exploration-r76-joint/tables.md)为100条件/20均值，原80条件与八门槛原样保留。ND空间PSNR较好但未建立跨条件时间优势，见[R75决定](research/4.temporal-consistency/docs/exploration-r75-decision.md)和[R76决定](research/4.temporal-consistency/docs/exploration-r76-decision.md)。current/self4是同权重输入控制，非独立训练空间模型；四次self4非同计算。

这只是发布ReBotNet-XS的自启动/原生分块任务适配，MobileIE8K/128与XS64K/384训练条件不等，不是原论文500K复现、原创方法或SOTA。SYN480×854和ND288×512原生裁块均重复DEV，非整幅1080p/自然夜间/手机/独立最终。完整证据现已接入上述build020；旧STOP与保留像素保护保持。以下启动/资格叙述仅为历史，不据此重开已结束任务。

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
