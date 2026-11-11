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

10. 03:05：26个wheel共151529343字节全部匹配官方SHA256，离线安装成功。新尝试 `attempt-20261112T030521` 再次通过无活动实验屏障和包归属检查，安装锁定清单单独提交。
11. 03:06:17：迁移完成。缓存2106个常规文件，共9269917305字节，迁移前后大小/内容SHA256全部相同；仓库 `.cache -> ../../data/MobileIE-Neo/cache`、`.venv -> ../../data/MobileIE-Neo/.venv`。原155MB环境移入自有非活动归档，移除shared-torch.pth并禁用system-site-packages，没有修改其他用户文件。
12. 最终验证：所有研究包真实路径位于自有环境，缓存变量位于自有资产目录；既有video_15共16帧CPU回放的四项指标差0，未初始化CUDA。通过新的 `./scripts/project-python` 再次核对环境前缀、包文件、两个Git链接mode120000和仓库剩余忽略资产，无遗漏运行数据目录。各清单、验证结果、日志和两个链接逐文件提交。

迁移现已实际完成；首失败尝试保留，最终证据见 `completed.json`、新attempt的 `07-final-verification.json` 和 `09-entrypoint-and-layout.json`。阶段推送完成后，使用 `./scripts/project-python` 重新经共享GPU空闲双采样启动seed1，预算仍为post_transport两目标各4000步。
