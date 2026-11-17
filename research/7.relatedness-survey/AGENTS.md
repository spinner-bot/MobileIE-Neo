# MobileIE相关性调研工作站边界

创建：2026-11-17T18:20:36+08:00；作者：Codex  (sqs_A100) <langxibielangle@qq.com>；触发：用户明确要求文件架构独立，降低后续合并冲突。

- 本任务仅在survey/pre及本目录树内新增／修改版本化文件；不向仓库根README、根AGENTS、全局docs索引、既有research/5.prior-art-survey或research/4.temporal-consistency追加本任务记录。既有specs只读。
- 本目录README为独立入口；gates、collection、resources、docs、report、scripts均在本目录内。进展编号在本工作站独立命名，不抢占主线或演示的报告编号。
- 已冻结Gate标准路径／内容保持不变，按gates/freeze-v1.json核验；过程和新增决策另写本地docs，不改冻结文件。
- 仓库外原始大资料限zrh自有data/MobileIE-Neo/cache/relatedness-survey命名空间，以本目录archive相对链接进入；不覆盖其它实验的缓存／下载／检查点，不修改共享环境。
- 只读引用既有项目资料，记录来源提交／路径，不为调研修改主线算法、活跃依赖、演示实现或资源调度。脚本自包含于本目录，从自身路径定位工作站；避免要求改动项目公共配置或依赖清单。
- 未来合并保留本目录边界；同步仍为fetch及必要时显式merge --no-ff，不自动引入main／demo的新提交，不rebase、squash或强推。Git提交仅stage本目录明确文件；使用-c gc.auto=0避免密集提交重复触发共享仓库自动整理，不改公共Git配置。
- 本任务长期目标、Gate顺序、证据受限结项与作者／邮箱等要求见docs/0001-任务与决策-26111718.md。本约定不会解封既有最终数据或启动Gate3实验。
