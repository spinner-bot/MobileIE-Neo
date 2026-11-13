# OLiVES既有保留来源：元数据资格，像素未解封

2026-11-13；Codex  (sqs_A100)。不是已完成独立最终测试或新取得视频。延续R9的保留ID，不改变现有训练/开发划分。

来源固定作者[OLiVES-Lowlight/OLiVES](https://huggingface.co/datasets/OLiVES-Lowlight/OLiVES/tree/291ae19d061dedcff8e62f9b6d9d33d7ceeedbaf)；已有作者清单96 train/41 test。该固定revision当前完整一层API目录为40 GT/137 input，双侧ID实际31 train+9 test；项目已使用的6个作者test是探索资料，不能重新称未见测试。另12训练及4 DEV来自作者train，角色保持。

|既有保留ID|作者角色|共同PNG帧名（仅元数据）|与736个既有资产SHA精确重合|PNG/视频读取|
|---|---|---:|---:|---|
|video_43|test|200|0|未读取|
|video_71|test|225|0|未读取|
|video_83|test|161|0|未读取|

八个固定revision目录请求正常，共383301字节JSON，均无next Link；没有PNG/缩略图/viewer/resolve请求、图像解码、flow/cache生成、CNN/SGD/GPU。每片两侧全部已列PNG的LFS oid参与检查；共586对帧名是目录覆盖，不是已下载/解码的586帧，也未实测其shape/帧率或标定。

[资格结果](../../process/olives-reserved-metadata/qualification001/summary.json)、[目录身份](../../process/olives-reserved-metadata/qualification001/responses.json)、[候选元数据](../../process/olives-reserved-metadata/qualification001/candidates.json)、[范围方案](../../docs/olives-reserved-metadata-protocol.md)均保留。原始API文本仅在自有data的cache/datasets/OLiVES-reserved-metadata/qualification001。候选文件中的首16名称仅沿既有登记惯例留元数据指针，**不意味着最终评价已经决定只取16帧**；正式时序长度/全视野/采样/时间显示条件仍须预先冻结。

零SHA重合仅排除已登记资产中的逐文件重复，不证明不同物理场景/拍摄take、作者预训练未见、全网/未登记访问不存在或可直接下载。R3 example1与旧video21的32项重复已包含在736个唯一已取得SHA中，没有重复当新来源。当前canonical保留PNG缓存目录仍不存在，不扩称服务器上所有可能历史访问均已审计。

继续保留三段像素，不用于DEV调参、选权重或新网格；R75/R76/R77即使过其门槛也不自动解封。只有确立最终方法并冻结训练权重/预处理/评价/完整参考及读取协议后，才另行决定实际最终测试。本来源仍为作者ND实拍设置，不能补称自然夜景或手机成本；DAVIS test-dev30与其他保留像素要求保持。
