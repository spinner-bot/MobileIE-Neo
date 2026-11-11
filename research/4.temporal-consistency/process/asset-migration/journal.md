# 自有资产迁移过程

作者：Codex  (sqs_A100)；时间均为 +08:00。目标不是搬用共享环境，而是从公开依赖建立独立环境，将既有自有运行资产移动到 `/data/Users/zrh/data/MobileIE-Neo`。本记录随实际完成阶段追加。

1. 02:05：停止尚未准入、零更新的 R28 seed1 等待器。seed0 保留固定 4000 步预算，正常结束后才迁移。
2. 02:09 起：分别提交路径归属规范、根目录约定、依赖清单、迁移/验证代码和启动入口。公开来源预检确认所需版本可取得；未复制其他用户环境。
3. 02:20:35：首迁移尝试确认 `/data` 已挂载且项目虚拟环境实验已退出，创建自有 `.venv`，禁用 system/user site-packages。pip、临时文件、Torch 等缓存全部定位到项目资产目录。
4. 安装阶段：从 PyTorch 官方 cu126 索引安装 Torch 2.9.1+cu126 及依赖；另向新环境安装仓库版本的 `sitecustomize.py`，并记录该补充动作。首次驱动加载版本与后续源码差异在 `source-at-launch.json` 明确保留。
5. 02:38–02:40：单独提交 seed0 完成状态、研究收口、论文证据表与新入口文档；SSH 推送至 `2221496`，共保留 205 个小步提交。推送成功回执已另行提交，未把仍在安装的迁移标记完成。
6. 约02:47：Torch/CUDA 依赖安装完成；研究依赖进入 PyPI 下载。NumPy 随后发生读取超时，续传后仍约每秒数十KB。检查官方 CDN 与公开镜像的连接，仅用于诊断，没有从镜像安装依赖。
7. 02:55：只向已核对命令的 pip 安装子进程发送 SIGTERM，结束低速传输；没有截断任何 GPU 实验，没有移动缓存或切换链接。首尝试的失败、安装日志和驱动日志分别提交；保留全部事实，不覆盖旧日志。
8. 恢复方式：系统 Python 运行 `scripts/prepare_owned_wheelhouse.py`，读取首安装日志中已解析的 wheel 文件名，从官方 PyPI JSON 获取 URL、大小及 SHA256，最多四个256KiB分块请求下载官方文件。完整摘要匹配后才形成可安装 wheel，存放在自有资产目录 `wheelhouse`。记录明确区分 partial 和 complete。
9. 后续恢复流程已启动：仅在 wheelhouse 完整且每个摘要复核通过后，使用自有 Python 执行 `pip install --no-index --find-links /data/Users/zrh/data/MobileIE-Neo/wheelhouse -r requirements-research.txt`。然后重新运行迁移脚本，使用新的 attempt 目录和 `execution-retry.log`；须通过包路径、缓存清单、链接、既有CPU参考回放后才产生 completed 标记。

当前第9步尚未完成，不据本记录宣告迁移成功。完成记录、最终清单、实际链接、安装锁定清单和核验结果须逐文件提交并阶段推送，随后使用 `./scripts/project-python` 重新经共享GPU空闲预检启动 seed1。
