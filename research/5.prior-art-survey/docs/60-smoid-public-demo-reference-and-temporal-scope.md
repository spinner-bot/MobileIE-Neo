# SMOID公开demo：参考身份与时间接口

2026-11-13；Codex  (sqs_A100)。补充note29的准备度，针对外部配对验证缺口核查一个既有作者入口；没有新增模型或下载视频像素。[有限方案](../../4.temporal-consistency/docs/smoid-demo-source-qualification-protocol.md)与[实际资格记录](../../4.temporal-consistency/process/external-smoid/qualification001/runtime.json)分开保存。

固定[作者仓库53e81c4](https://github.com/MichaelHYJiang/Learning-to-See-Moving-Objects-in-the-Dark/tree/53e81c4eb8862d4d706da43aa007fa16f3f09610)，六份公开文本逐项Git blob与本地SHA正常，缓存仅自有data/SMOID。本次GitHub连接器传输失败，使用公开原始文件和普通GitHub元数据，不绕过权限。没有运行作者TensorFlow/Python2代码或安装旧环境。

[customized_list](https://github.com/MichaelHYJiang/Learning-to-See-Moving-Objects-in-the-Dark/blob/53e81c4eb8862d4d706da43aa007fa16f3f09610/customized_list)三行均给相同input/GT路径：Cam1 gain10、Cam2 gain15、Outdoor gain10_15fps_2。结合[test.py](https://github.com/MichaelHYJiang/Learning-to-See-Moving-Objects-in-the-Dark/blob/53e81c4eb8862d4d706da43aa007fa16f3f09610/test.py)只加载input并导出预测的实际路径，这些是demo占位清单，**不能用第二列自动构造独立参考**。不能由此断言论文完整数据没有GT、另两例包失效或整个ZIP绝无参考；本次没有查看归档目录或像素。

test的输入为四通道packed RAW，输出为两倍宽高的三通道结果；[train.py](https://github.com/MichaelHYJiang/Learning-to-See-Moving-Objects-in-the-Dark/blob/53e81c4eb8862d4d706da43aa007fa16f3f09610/train.py)分别按65535和255归一输入/GT，不能用现有8bit RGB接口直接替代。`process_video`先对传入完整片段的全部RAW数值作直方图均衡，再使用32帧窗口；长片按MAX_FRAME800分段。由此推得demo输出一般依赖片段内后续值，不能称因果单帧参考。这里是代码推论，不推广为原论文所有配置的严格时延结论；没有模拟指标或运行网络。

[下载脚本](https://github.com/MichaelHYJiang/Learning-to-See-Moving-Objects-in-the-Dark/blob/53e81c4eb8862d4d706da43aa007fa16f3f09610/download_dataset.py)提供三例包。只对Cam1固定ID做一次普通匿名HEAD，实际返回404、text/html，0 body读取，未取得归档。按方案停止，未请求另两包、换镜像、登录或重试。404只说明该URL/当次请求不可用，不证明作者永久删除或其他访问方式必定不可用。

结论：这条已公布demo路径尚不能增加配对外部基准数，不下载大包来猜GT、不把输入本身当参考。现有来源仍只能确认作者文档所述ND/双机实拍接口，不能补称自然暗夜连续配对。原R75训练/R76补充等待及其17/27/37冻结源不改，主表没有新PSNR/JOD或方法成功；外部自然动态、手机与独立最终验证仍需实际解决。
