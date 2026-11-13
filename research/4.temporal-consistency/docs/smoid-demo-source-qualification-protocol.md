# SMOID公开demo的有限来源资格

2026-11-13；Codex  (sqs_A100)。目的为确认另一个实拍视频入口是否能补配对外部验证，不开展模型/参数网格，不改活跃R75/R76源或门槛，不读取保留test-dev30。

固定作者仓库MichaelHYJiang/Learning-to-See-Moving-Objects-in-the-Dark的提交53e81c4eb8862d4d706da43aa007fa16f3f09610。只保存README、download_dataset.py、customized_list、test.py、config.py、train.py的公开原始文本，每份≤64KiB、无重试，核Git blob身份；文本缓存仅在zrh自有data/SMOID下。不会执行作者Python2/TensorFlow脚本、安装旧框架或下载模型。

已读customized_list三行均将同一.npy写入input/gt两个槽，尚无可确认的独立GT；这是demo清单事实，不能推为完整数据集不存在GT。test.py按完整输入片段作直方图均衡、使用32帧窗口并可按800帧分段，不能把demo直接当单帧/因果RGB基线。源码阅读不等于完成论文全部实验复现。

只对下载脚本固定Cam1例包ID 1qES6teQUprs-cgL-nACzFighjm_Dho0L做一次普通匿名HEAD，0 body读取、0图像解码，不访问另两包或未知镜像；权限/配额/网络失败保留并停止。HEAD可用不等于有独立GT，也不自动获取ZIP目录、数组、缩略图或像素。后续若目录资格必要，须另定传输范围；没有独立参考身份则不构造PSNR/JOD或把自身输入当GT。

所有代码、source manifest、正常/失败终态、解释及过程分别提交，阶段推送。R75/R76原实际进程继续；这不增加已完成外部基准数，不解决自然夜景/手机/独立最终缺口。
