# R52 Night Wenzhou作者样例获取

记录：2026-11-12；作者：Codex  (sqs_A100)。资料/协议/脚本/获取状态/过程/报告分开提交；所有下载只进zrh项目data。

定向读论文v5 III-D/Table IV，核查作者库固定98d778d5b6686ba2d4318f566ef9d7e3f6403551。原完整Drive ID 1kRsszLDx1KKx6XHeLFfogDJUO62GNLAc匿名404，未反复重试或使用未经证实镜像。最初只读issue1不足以判断全部issues，协议措辞已纠正，随后实际读取全部issues1–7及相关作者评论。

issue6作者说明原学校账户失效，并提供部分帧/视频备份；issue7仍在寻找完整集合。公开备用ID 1bIaPaFeOWDwuhWgLXraY1FrSZBAqJF2T正常匿名大文件确认后HEAD200、application/octet-stream、1170848021字节。正常确认表单的临时URL/UUID不保存；不登录、不绕过配额、不联系作者。

19:36:57实际启动下载PID3514885，单响应流、每1MiB写自有.cache/datasets/Night-Wenzhou/author-samples/attempt001/Wenzhou-Night.rar.part。仅完整长度/RAR签名/流式SHA均通过才rename并写manifest；SHA为本地校验，不是作者提供。日志每64MiB报告进度。下载尚在进行，不能把partial当成功或说已完成夜间验证。

19:58:10启动finalize001 PID3531976，绑定下载进程身份与三个脚本摘要，不重启下载、无墙钟截止。只在下载正常终态后用系统libarchive列目录，保留库版本/成员类型/原始字节文件名。RAR跳过solid数据可能内部解压，但不提取成员、不调用图像视频解码器。失败保留过程，不能自动重试或改正在执行的源码。

归档被明确标为部分作者备份，不是26.144GB完整基准、配对GT或手机实测。之后需先看目录分清原始低光输入与作者增强输出，再独立登记固定片段/处理/模型名单；当前0帧解码/0CNN/0GPU。
