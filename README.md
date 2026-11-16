# MobileIE-Neo

面向手机端的轻量图像增强模型研究：在 MobileIE（极轻量实时图像增强 ConvNet）基础上，解决其应用于视频时产生的帧间闪烁问题。

## 当前研究状态

最新[R82 / 0161](docs/进展报告/0161-T-26111707-自参照修正四万八千更新完成与完整评价接续.md)：双seed三控制各8000、合48000正式更新已全部正常完成，六最终重载差0；原训练PID退出/GPU释放。原协调接续CPU971166/971167，41源/两协议保持，完整100新质量/JOD、24比较和20图仍待，不根据前缀判断收益。训练完成非方法成功；旧STOP/保留最终/paper020与论文缺口保持。

此前R82 CPU资格[R82 / 0159](docs/进展报告/0159-T-26111706-核预测先例收口与自参照修正原型资格.md)：KPN/BPN/F2R直接先例已限定；304参数共同条件修正头通过真实训练窗CPU资格（12丢弃更新、0正式/DEV/GPU），self/流式差均0、过去梯度非零但很小，非历史有效性证据。双seed三控制共48000正式更新及完整共同评价已预登记；完整实现与CUDA资格尚待，无正式作业。旧STOP/最终像素/paper020保持。

截至2026-11-17，项目仍处于方法探索阶段，尚未建立可投稿的新方法。接续先读[进展报告0161](docs/进展报告/0161-T-26111707-自参照修正四万八千更新完成与完整评价接续.md)、[4号研究入口](research/4.temporal-consistency/README.md)和[服务器约定](AGENTS.md)。主线在 `main` 推进，移动演示由独立支线处理。



最新R81（[0158](docs/进展报告/0158-T-26111704-训练噪声门控干预与局部路线止损.md)/[停止决定](research/4.temporal-consistency/docs/exploration-r81-decision.md)）：一次固定训练图块静态噪声干预已正常结束，8主条件0/8。128唯一噪声图由双seed共用、272骨干/1040tail、0SGD/DEV/GPU；8固定图均已实际查看。无噪参照localgate有部分PSNR/7.5%–9.2%波动收益但不稳，global替换可能与当前特征失配，均非部署/最佳门控上界。不继续此小门控/温度/滞回/EMA等网格；R80及全部旧STOP/最终像素/paper020保持，尚无新方法。

此前R80已完整收口（[0157](docs/进展报告/0157-T-26111703-有限前缀条件读出完整负结果与先例边界.md)/[停止决定](research/4.temporal-consistency/docs/exploration-r80-decision.md)）：160新质量及160官方JOD、各6368比较帧，联合200条件/40均值/24比较通过4，tail0/12，20固定图已逐张实际查看；原CPU/指标/协调全部正常退出、31源保持，无活跃R80作业。tail相对raw时序仅降.81%–1.88%，对强rec仍高38.13%–50.96%，停止本13basis/ridge/暖机/teacher/seed局部网格。GT诊断ND PSNR虽明显改善但时间几乎不降，非部署结果；note65核少量通道适配/线性展开已有先例。尚无有效新方法，旧STOP/最终像素保护及paper020保持。

当前[R79完整决定](research/4.temporal-consistency/docs/exploration-r79-decision.md)：24000更新、100新质量/4480帧与100 JOD均正常结束，联合140条件/28域均值/16比较全部False，全部20固定图已实际查看。课程相对同权重current时间代理仅降.48%–.83%，相对强rec仍高32.68%–51.74%；正负条件/seed都保留，按协议停止本课程局部网格，不推广为所有课程无效。原CPU/感知/协调均退出，禁止据旧快照重启；21/11源未变。[note63](research/5.prior-art-survey/docs/63-fip-diagnostics-and-curriculum-novelty-boundary.md)核明FIP训练/诊断已有先例，非创新。完整结果暂独立、paper020未变；有效新方法、自然动态/手机/独立最终仍缺，保留像素/旧STOP保持。

前序定向核查[训练信息课程先例](research/5.prior-art-survey/docs/62-training-information-curricula-and-photometric-transfer.md)：FIP遮中心预训练/LVD不平衡噪声已有作者代码，不能直接作为创新。R78仅人工光度资格（8行、92/92），显示把暗输入−亮GT当噪声缩放会在无噪时也改变曝光；真实暗clean/噪声与课程有效性未被识别。该人工资格已关闭、不再追加同类网格；实际网络学习另登记为上述R79。

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
