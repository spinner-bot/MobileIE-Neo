# CRVD官方训练资料的有限获取资格

登记：2026-11-13；作者：Codex  (sqs_A100)。本阶段独立于活跃R73，不修改其源码、预算、输入或门槛，也不评价任何候选。

目的：把已确认的官方真实RAW噪声来源从README声明推进到可核查的训练资料准备。CRVD室内序列由静止位置的多张采集平均并作BM3D生成参考；不是自然连续运动夜间配对。未滤波平均与BM3D参考须区分，不能称精确无噪声真值。原文§3.1及补充§1.1/表1、原作者README为来源，详见既有external-raw-video-readiness记录。

阶段A只获取Google Drive官方公开目录HTML，不取TIFF、缩略图或其他图像。入口固定作者链接`1TZhNrsJ4tmKaU7b9b_9hkJIhwMi9Tfk3`，只沿indoor_raw_gt与indoor_raw_noisy，分别到scene1/ISO1600；最多七个串行目录请求，每份HTML≤2MiB。它们是当次可见条目的目录，不宣称取得整数据集清单。其他训练scene、测试scene7–11、outdoor均不遍历图像内容。目录结构不符、需要登录、配额或网络失败时保留失败并停止，不改从未知镜像取得数据。

已有一次标准库只读RAM入口探测成功：HTTP200/text-html、公开_DRIVE_ivd存在，329834字节，SHA256为`0aca825db0299149967735f1471ed2cba6e7244560c8c59c24e32a1e7c6560af`；未保存或读取任何像素。现在代码先提交，再按新目录保存可复核来源。仅项目自有Python标准库，无新环境/依赖。

目录正常后，根据实际列出的稳定文件ID，另登记scene1/ISO1600/frame1的有限输入资格：固定两个噪声重复和原始平均/BM3D两种参考，精确文件名/大小/上限在任何二进制读取前写入选择清单。若官方目录不提供所需身份，则停止在准备度，不猜配对或打开其他场景。是否扩成后续噪声研究/固定RGB渲染须另有明确问题；不自动运行训练、CNN或GPU，不把资料资格当完整基准结果。

目录HTML、后续原始资产只保存于`/data/Users/zrh/data/MobileIE-Neo/cache/datasets/CRVD`；仓库保留代码、路径/摘要、目录记录、正常/失败runtime和过程。写前检查/data挂载，分别小步提交并阶段推送，指定作者/no-ff保持；原R73训练及CPU评价继续。

## 实际发布布局修订（08:47 CST，二进制仍未读取）

catalog001取得根HTML后因记录路径转换错误退出，0图像；已保存其代码版本/配置/异常/runtime，修正为仓库现有.cache入口。catalog002随后正常读根与GT目录，发现真实发布为`indoor_raw_gt_scene1.zip`等11个场景ZIP，按注册结构检查停止，未猜新的层级或配对。

因此阶段A改为三个串行公开HTML请求：根、indoor_raw_gt、indoor_raw_noisy；只选择各目录中与`{目录名}_scene1.zip`完全同名且MIME为application/zip的一个档案，保留全部当次可见元数据。仍≤2MiB/页、不取图像，不遍历其他场景内容。新编号catalog003，不覆盖既有失败；这是实际发布格式修正，不变更训练场景或引入测试像素。

下一阶段先对两个选定公开训练档案作只读HEAD，确定可访问性/长度；若允许，再单独登记有界ZIP目录Range与四个固定成员的获取，不能把HTTP200目录等同完整资产可用。配额/权限问题不绕过，不扩其他数据主机或场景。原文与补充参照[CVPR论文](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yue_Supervised_Raw_Video_Denoising_With_a_Benchmark_Dataset_on_Dynamic_CVPR_2020_paper.pdf)、[补充§1.1/表1](https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Yue_Supervised_Raw_Video_CVPR_2020_supplemental.pdf)：当前ISO1600平均150帧，参考并非独立无误差真值；七个手动静止位置不当作连续20fps自然运动。

## 阶段B目录登记（08:52 CST）

catalog003正常3页，scene1 GT/noisy档案ID已固定；head001正常返回200、application/octet-stream、Accept-Ranges=bytes，长度分别130912326与1230174943，0 body读取。阶段B只取各ZIP最后65536字节及中央目录，每目录≤4MiB、单请求≤1MiB，复用既有SplitArchive的严格206/Content-Range/长度检查；无全档案下载或图像解码。尾段可能包含压缩图像字节，明确计入传输量，不称0图像字节。仅网络技术故障至多三次，权限/配额立即停止，不换来源。

中央目录正常后按实际原始成员名固定scene1/ISO1600/frame1：GT档案clean、noisy档案原始平均clean及noisy0/noisy1。若精确成员不存在，停止并保留目录，不更换场景/ISO/时刻。获取前另写四成员大小、CRC/偏移/压缩方式清单并提交；仅在清单已冻结后取对应压缩数据，合计传输上限32MiB、单文件解压≤8MiB。资格仅原生TIFF容器、uint16/尺寸/值域与参考类型，无渲染、调亮、CNN/SGD/GPU或新模型评价，不作完整噪声标定/独立GT或夜间连续视频结论。

## 四成员的原生格式资格（目录已正常，输入尚未读取）

GT原始成员实际名为`scene1/ISO1600/frame1_clean_and_slightly_denoised.tiff`，noisy档案另有同位置`frame1_clean.tiff`和`frame1_noisy0.tiff`、`frame1_noisy1.tiff`；只用这四项，不改变场景/ISO/时刻。官方test_indoor.py同样读取带slightly_denoised后缀的原生TIFF，再转float并使用240/4095归一；本阶段不执行该模型或归一。

目录显示前三个noisy侧成员各4147322字节，BM3D参考8294534字节，后者仍≤8MiB；因此不预设四项都为uint16。传感器噪声及平均参考保留uint16，轻度去噪参考保留实际TIFF原生数值类型并检查有限值/1080×1920形状，记录其值域而不按假定范围裁剪或转换。每个单请求无技术重试，源CRC/大小与实际解压核对后才产生完整TIFF，四项32MiB传输总上限保持。两个噪声观测与平均参考可能共享采样，不能称与GT独立。
