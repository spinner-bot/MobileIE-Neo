# MIE-0023｜FJNDF-Pytorch项目分析

2026-11-21；Codex  (sqs_A100) <langxibielangle@qq.com>。有效序26，MIE0022关闭后开始；本项分析同一固定版本的项目载体，复用证据透明列SHA，不算新增独立科研实证。0研究源码执行/模型加载/训练/编码/指标/GPU/Gate3。

## 固定载体和真实阅读

commit3b46739e4f524c205c9c277f9c4b1673b9d5aedc（2026-11-08T11:09:08Z）、tree8164518198b68bb06ddd25c0cacc8c67c57f34a9；150文件2121784B与MIE22及原ZIP/官方blob一致，128文本链接不是全部逐行阅读。46适用全文和明确部分函数的已完成静态阅读、参数/损失/权重/数据/结果核查按12份不可变证据SHA复用；新增完整读regr_filt/run/makefile/regBD/sampler/datasethelper等项目入口。当前未用经典模型全部数学、通用resize/logging等不伪称全读。

所有3SVG实际离线渲染/查看，原字节不变，字体替代造成部分文字重叠是本地衍生限制，不归咎作者。基准图展示静态工具/训练/编码链，训练图展示MBR/FST/HDPA残差与两参照loss；JND图虽列TM，代码实际只是ones占位。kodim01.yuv仅768×512输入Y平面显示石墙/窗户，无预测/GT/视频，不作效果证据。原150工件保留，包括3PTH；重新CRC核，原操作码阅读复用，无反序列化/模型加载。

README项目citation写Chenlong He，commit显示nanushio/NanUshio，只记录各自角色，不推断用户名与真实作者对应，也不把机构owner当作者名单。关联论文MIE22四作者/v3状态与项目引用身份分开。

## 核心功能及先例

作者论文直接链接本repo，当前模型文件明确继承MobileIE。Lite默认Y单通道残差网络，UV回拼，8/8/12通道、FST平方和当前帧全局/局部attention。折叠1861/训练24949的源算术与论文相符，完整wrapper仍保留训练网+slim，统计脚本范围不同；基线5×5首层41413/3257与论文38.92K/3.06K版本不合，假想3×3解释仍只是线索。发布权重首conv字面支持当前kernel，不证明所有论文历史训练或表内模型映射。

Conservation实际线性hinge与论文平方不一致；默认K10低频集合却相同，不把非交替遍历误判成默认mask错误。全系数正交DCT residual的Parseval等价只限该项，约束的符号/相位/时间边界保留；pred单边softclip与验证uint8保存需单列。这些是具体训练/部署重复先例，不能用“MobileIE+频率/感知”组合本身作视频创新。

## 推理入口、配置与完整pipeline

单文件入口每帧新task并加载权重；批处理每video一次加载、逐帧复用。均无跨输出hidden-state/时间损失，使用视频文件不等实现时间去闪烁。pad32/crop和YUV全量程归一化、UVrepeat/抽样、保存clamp+round都在网络之外。单文件有Y4M分支，批处理参数收集对Y4M/image的continue使其不加入任务，不说整个库无此支持。

get_model仅显式传cfg的name/precision/weightpath，具体结构从DEFAULT_CONFIGS指向的固定effect_yml_path读取；修改任意其它临时cfg字段不保证传到模型。pretrained_model_path字段存在就触发加载，pretrained布尔未独立控制；默认提供配置True/固定路径可对上，不把自定义场景问题称默认失败。CLI target最终由FrequencyJNDModel的cfg/self.target决定，不能仅改命令行就假定改变注入。

默认fp32；半精度选项下新建slim默认dtype后只to(device)，没有可验证的统一FP16导出证明，需独立控制转换/折叠/边界误差。源码未提供手机TFLite/ONNX/NPU部署工件或完整成本链；不能由1.86K或图像GPU计时推手机持续视频FPS。

## 训练、数据与评估

固定yaml100K/b16/center224、seed123、cosine/warmup/weightdecay和split_num1；paired flip/rot及当前Y损失路径已复用读。row-random split生成器不保证scene/clip分组；28996B实际PKL只操作码/范围0..2007检查，未重构具体phase成员或认证实际训练使用。所有2008训练名与5/15/30/50外测元数据已留；数据文档KonIQ/KonJND歧义、原数据许可/语义重叠未知。dataset_api helper主要提取预期本地archive并构建dataset，没有实现文档所暗示的实际网络下载；README的手工准备流程单独存在，未运行。

验证teacher目标与编码ori目标不同，各指标best可不同迭代，不能把多项best拼一个检查点成绩。只报告接口存在和作者结果，不认证代码训练再现论文。原模型、Lite、IQNet在此框架按JND目标重新训练，不代表原任务原权重表现或同原论文部署条件。

当前regr_encode与regr_calbdrate只有ori激活，其它方法全注释；原样运行后者是ori对自身，不能当复现完整表。编码默认仅一帧，故周期200参数在该条件不与首帧intra矛盾，扩多帧则须重新冻结GOP。8bit420断言与VMAF固定设置相符，不能当默认格式bug；不同质量指标有不同裁边/模型/汇总口径。BD整体先均值RD再Akima、两侧独立剔无效行，共同样本完整性未知，不是每视频BD直接平均。

当前16组表/32个baseline-Lite行与论文部分表相对应，ALL不同列数须分开，公开大模型行实际是BaseLine+Lf。原曲线/逐视频结果还未取得，不能认证每个样本/缺测或重算BD。当前手机/连续视频/功耗/峰值内存未知。

## 获取、许可与复现边界

README的Torch2.4.1等环境说明不锁定其余requirements；makefile克隆codec的远端分支且没有固定revision，现有文件不能重建作者二进制完整版本。未安装/编译/运行它们。保留FDU非商业许可与README CC BY-NC-SA4.0声明，不替代各原数据/第三方组件许可认证。

作者逐视频结果分享仍可正常公开码访问并显示results.tar.gz1070637B，MIE22已完成HTTP/普通浏览器多途径尝试且实际点普通下载，但无实际文件，未知真实原因，不能称不存在/已读或编造登录付费拦截。本项同SHA继承完整获取记录，没有重复执行网页流程。Git150件完整不等所有平台外补充已读。因此建议证据受限结项：可得适用源与示例完成静态分析，公开宣称的逐视频补充数据仍缺内容。合法取得后追加复审，旧结项不改。

## 二期用途与控制

本项目适合作为伪teacher/约束训练、亮度通道压缩及真实编码评价工具链先例。二期须明确RGB/亮度/teacher/数据/历史未来信息预算，分离空间压缩指标与连续可见闪烁，固定原始曲线/公共样本/序列划分，先对齐模型和工具链版本，再评价相位/曝光/真实运动噪声与全手机预算。所有建议交接未来实验分支，当前不执行预实验，不提前作全部资源综合结论。
