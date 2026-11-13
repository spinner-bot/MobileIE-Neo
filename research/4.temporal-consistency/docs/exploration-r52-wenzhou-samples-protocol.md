# R52｜作者备用Night Wenzhou样例归档获取，未进入评价

登记：2026-11-12T19:36:10.728053+08:00；作者：Codex  (sqs_A100)。原完整26GB链接404，但[作者issue6](https://github.com/ShenZheng2000/LLIE_Survey/issues/6#issuecomment-2597100240)明确给出部分帧/视频备用公开链接，issue7尚未恢复完整数据。本阶段只获取与列目录，不按模型结果挑视频。

稳定Drive file ID为1bIaPaFeOWDwuhWgLXraY1FrSZBAqJF2T，正常匿名大文件确认表单HEAD200，application/octet-stream，文件Wenzhou Night.rar，1170848021 bytes。允许按正常公开确认表单下载一次到zrh自有data；不保存临时表单查询URL/UUID，不登录/创建账号/发消息、不绕过权限/配额。如果返回HTML/权限/配额就停止记录。使用单响应流保存，验证完整长度/RAR签名、流式SHA256，失败保留partial与独立attempt，不将不完整文件当成功。

获取后只列归档目录（系统发行版libarchive由本项目标准Python ctypes只读调用），记录库版本/文件名/大小/文件类型。RAR可能solid，列目录时库可能读取/解压跳过压缩数据，但不逐成员提取、不调用任何图像/视频解码器；不能笼统称“未读取任何压缩像素字节”。归档与完整索引资产存自有data，元数据/manifest/过程入Git。

这只是作者备用样例，不是完整Night Wenzhou26GB或标准全基准；不含已知GT，不算新的配对PSNR测试/手机实测。未来只有在目录明确原始输入视频、来源可追溯时，另登记固定选片/时间段/原亮度全FOV处理及模型名单，才解码或推理。不能用作者已增强输出混作原输入；不能凭归档存在宣称已经完成真实夜间验证。现阶段0CNN/0GPU、不改R47冻结代码或最终测试。源码、获取、清单、过程/报告分别密集提交，阶段推送。
