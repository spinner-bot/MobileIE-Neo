# DID真实相机视频资料的当前准备度

记录：2026-11-12T05:57:34+08:00；作者：Codex  (sqs_A100)。触发：R34等待完整动态训练池与GPU期间，定向核验一个标准LLVE公开来源，不扩展候选模型/参数网格，不读正式测试像素。

## 来源与划分

[官方仓库](https://github.com/ciki000/DID/tree/6a14908df2d789858bf79155a24d86bc73742dcc)已固定commit。官方train_list.txt的blob4aeb4038b9056f17cdc2bdfdaa9c43e1352e27d7与test_list.txt的blob0e3662d580fe739c346be5c7368c8bb9eb3130e9分别得到238/87唯一ID、交集0、合计325。[版本化元数据](../data/external-did/official-split.json)保留原条目；与论文声明413对总量不同，尚未拿到ZIP目录核对，不能把325推成实际总量或补造缺失ID。

[作者项目页](https://buptmmlab.github.io/publications/2023/dancing-in-the-dark-a-benchmark-towards-general-low-light-video-enhancement/)描述五种相机、逐对原地切换ISO与小角度云台运动。本次只读该页与论文检索段落，未精读其全文；不把该来源自动称为同步双相机自由手持/自然夜景数据，实际曝光、噪声及空间/时间配对仍需后核验。

## 作者链接的匿名获取状态

README公开百度分享可经作者公开提取码正常验证，不需要用户登录即可取得目录。只列出DID_2k.zip（56650437780字节）、DID_1080.zip（32795182798字节）；标签不是已核实的真实帧shape/处理等价性。返回md5字段含非十六进制字符，按平台不透明字段保留，不能当标准MD5完整性证明。当前没有取得二进制/ZIP索引/像素，不能把HTTP200目录当下载成功。

作者在[issue2回复](https://github.com/ciki000/DID/issues/2#issuecomment-1858787035)明确提供OpenDataLab的ciki/DID镜像，故来源可核验。对官方文档推荐openxlab0.1.3包仅在自有cache读取PyPI wheel源码，未安装/导入、未修改环境；源码区分匿名info/list与需要JWT的get/download。第一次list请求超时没有当永久不可用结论；第二次有明确响应：id8134、公开状态、hasMediaFile=false、提示当前不提供直接下载。它与2024用户报告一致，但本结论来自当前实测，不拿旧issue代替当前状态。[目录记录](../process/external-did/public-directory-metadata.json)、[镜像当前回应](../process/external-did/openxlab-anonymous-metadata-attempt002.json)分别保留。

## 接续规则

准备度仅为来源/官方文本划分/分享目录已验证。DID图像、真实shape/配对/完整训练与测试、LAN代码及数值对照均未取得。后续有正常可访问归档时，先核验目录/大小，再另行登记训练侧开发片选择与共同预处理；官方87测试ID继续保留，不在开发时解码。不使用不明镜像、不借用其他用户本地资产，不因取数未完成改动R34既定训练预算。

本次没有扩大同类受限端点重试，也没有将HTML、目录或源码阅读包装成新数据实验。主线R32下载/R35CPU评价与R34等待协调器持续运行。
