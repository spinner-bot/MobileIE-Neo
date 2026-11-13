# MobileIE-Neo

面向手机端的轻量图像增强模型研究：在 MobileIE（极轻量实时图像增强 ConvNet）基础上，解决其应用于视频时产生的帧间闪烁问题。

## 当前研究状态

截至2026-11-13，项目仍处于方法探索阶段，尚未建立可投稿的新方法。接续先读[进展报告0103](docs/进展报告/0103-T-26111309-双种子训练预算完成与原生RAW资料资格收口.md)、[4号研究入口](research/4.temporal-consistency/README.md)和[服务器约定](AGENTS.md)。主线在 `main` 推进，移动演示由独立支线处理。

当前R73检验原生一/二阶历史矩进入第一次FST的固定组合机制：两seed、空间/均值/双矩三个控制，共24000次正式更新及六个最终导出均正常完成。首seed完整CPU评价正常，第二seed评价及随后感知/联合判定仍由原协调链接续；训练完成不代表质量门槛通过。按[已冻结协议](research/4.temporal-consistency/docs/exploration-r73-native-moment-core-protocol.md)等待完整结果，不依据部分分数追加局部网格。运行状态以原进程身份、实际命令和对应runtime为准，历史报告中的活跃提示不能作为重启依据。

R28/R34/R37/R44/R45/R48/R64等既有具体候选已按各自完整结果止损，不重开相似超参数搜索。R70传统视频滤波、R72公开REDS开发源强参考均已正常收口；这些开发证据不等于独立最终测试或自然夜间验证。CRVD仅一个训练时刻四份原生RAW文件的有限输入资格已完成，未进行模型评价，不计作完整视频基准。

现有[论文工作稿](paper/manuscript.pdf)为已审阅build015，尚缺可辩护创新、画质—时序—效率联合改善、自然动态配对、实际手机成本及独立最终验证。轻量移动端目标需要完整成本证据；原MobileIE的4075参数不是用户给定的硬性参数上限，含历史状态的模型也不能称零推理开销。

2026-11-12 03:06，自有环境和数据迁移已核验完成，两个仓库相对链接已入Git，后续统一用 `./scripts/project-python`；不重复迁移。详情见[迁移日志](research/4.temporal-consistency/process/asset-migration/journal.md)。代码、过程、结果、报告分别小步提交，阶段SSH推送，保持 `--no-ff`。新GPU任务须先检查共享卡的显存、利用率及计算进程，使用同组最多4张空闲卡。

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
