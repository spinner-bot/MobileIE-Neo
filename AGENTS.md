# MobileIE-Neo 服务器工作约定

更新：2026-11-12T02:09:53.875821+08:00；作者：Codex  (sqs_A100)。

- 本项目属于zrh。源码仓库为 `/data/Users/zrh/repo/MobileIE-Neo`；项目运行资产统一在 `/data/Users/zrh/data/MobileIE-Neo`。
- 禁止借用其他用户的虚拟环境、site-packages、数据或缓存；禁止在 `/root`、其他用户目录或全局Python中建立/安装本项目环境。
- 独立环境目标为资产目录下 `.venv`；仓库 `.venv` 指向它。仓库 `.cache` 指向资产目录下 `cache`，数据集、下载权重、训练检查点、运行缓存均经此入口进入自有数据目录。
- Python环境不得启用system-site-packages，不得以pth/PYTHONPATH引用其他用户目录。系统Python标准库及系统GPU驱动可只读使用。
- 后续启动使用项目自有环境入口；pip、Torch、Hugging Face、Matplotlib及临时缓存均设置到自有资产目录，不依赖默认/root缓存。
- 代码、版本化报告/实验记录和源码自带的小型资源随仓库管理；大体量运行资产不提交Git，路径、来源和迁移核验记录入库。
- 当前有已启动R28 seed0实验，保留固定4000步预算。结束后必须完成资产/独立环境迁移和链接核验，再重新启动已延后的post_transport seed1。不能只记录方案而遗留迁移。
- 写入前确认/data挂载；共享GPU启动前检查显存、利用率、计算进程，同组最多四张空闲卡，不以墙钟截断算法预算。
- main推进核心研究；移动演示支线不修改。Git作者使用 `Codex  (sqs_A100)`，保持文件/模块级小步提交，过程/结果/报告分别提交，阶段完成推送。
- pull/merge显式使用 `--no-ff`；不rebase、squash或强推。遵守[版本规范](specs/项目规范/版本管理规范.md)及[路径规范](specs/项目规范/路径与环境规范.md)。
