# CRVD原生训练资料资格

2026-11-13；作者：Codex  (sqs_A100)。[方案与修订](../../docs/crvd-source-qualification-protocol.md)、[冻结四成员](packet-selection001.json)、[完整取得与资格](../../process/external-crvd/acquisition001/manifest.json)。这仅是scene1/ISO1600/frame1一个训练时刻的两个噪声重复和两种参考，没有取得一个完整动态视频或进行模型评价。

来源为[CRVD/RViDeNet作者仓库](https://github.com/cao-cong/RViDeNet)交叉链接的官方Drive。数据作者为Huanjing Yue、Cong Cao、Lei Liao、Ronghe Chu、Jingyu Yang；论文为CVPR2020的Supervised Raw Video Denoising with a Benchmark Dataset on Dynamic Scenes。作者README标数据为CC BY-NC-SA 4.0、非商业学术用途。仓库只记录元数据与取得代码，原始字节存放自有zrh data。

|固定角色|原生类型|形状H×W|观察到的原始值域|
|---|---|---|---|
|BM3D轻度去噪参考|float32|1080×1920|249..4095|
|未滤波平均参考|uint16|1080×1920|244..4094|
|noisy0|uint16|1080×1920|215..4094|
|noisy1|uint16|1080×1920|215..4094|

四个原始TIFF共20736500字节，ZIP成员CRC/解压大小全部匹配；有界单次Range传输13500199字节、0重试，全部在预先32MiB上限内。两个档案的中央目录此前另传176208字节，尾段可能包括压缩图像字节，已按档案传输记录。OpenCV读出类型与原始TIFF IFD的BitsPerSample/SampleFormat核对一致，未作转换、裁剪、黑电平减除、demosaic、调亮或RGB渲染。源文档的GBRG/黑电平240/白电平4095分别保留为来源说明，没有由四幅图重新估计CFA或标定传感器。

噪声观测存在低于标称黑电平的值，原文件完整保留；不能先裁剪后再把它称为原噪声。平均参考与噪声观测的采样独立性未建立，轻度去噪参考也带人工滤波，两者都不能当精确独立无噪声真值。

当前只证明一个官方训练资料包可以取得并按原生格式读取，不证明噪声模型、时域性能、自然夜间或跨来源泛化。scene2–11和outdoor没有读取图像；未打开测试内容。后续若确需RAW噪声研究或共同RGB渲染，须另登记有限问题/资料范围与参考口径，不能自动扩为训练或替代R73的完整采用门槛。

过程保留：catalog001路径记录错误、catalog002按注册规则发现ZIP布局后停止；catalog003按实际发布形式正常取得三份元数据，head001两份头信息正常，archive-catalog001目录正常，acquisition001四文件正常。原始HTML、ZIP目录尾段/中央目录、压缩成员和TIFF均在`/data/Users/zrh/data/MobileIE-Neo/cache/datasets/CRVD`，不使用他人目录或新环境。所有阶段代码、来源、正常/失败记录分别提交；R73原训练与评价未改。
