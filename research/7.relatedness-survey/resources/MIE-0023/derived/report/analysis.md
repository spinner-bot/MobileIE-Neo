# MIE-0023｜FJNDF作者项目：完整源码载体与未取得的逐视频补充

**当前项目静态分析完成，建议closed_limited，待Gate2另轮审查。** 同固定150工件全部核验，实际阅读/图示/入口边界有记录，作者逐视频结果CSV内容仍未取得。0研究源码/模型/训练/编码/指标/GPU/Gate3执行。

[项目阅读](../docs/reading-notes.md)、[原件](../docs/author-source-manifest.json)、[同源证据对应](../docs/source-correspondence-audit.json)、[入口审查](../docs/project-entry-audit.json)、[实际示例/图示查看](../docs/visual-review.json)、[主要主张](claims.json)。

项目为MobileIE派生的小网络提供JND teacher/混合loss、YUV推理和编码BD链。它的时间相关名称和视频文件入口不代表时间模型：TM是ones，模型没有跨输出state，当前编码只处理首帧。单文件入口每帧重载，批入口每video复用；网络之外的IO/pad/颜色保存/codec开销需单列。

当前Lite的1861折叠参数有字面源支持，统计wrapper含训练和slim两套；baseline参数/首kernel版本、线性与平方loss、pred softclip和ALL列集的差异须沿用MIE22的明确边界。CLI/config默认不等任意自定义选项生效，原样回归脚本只启用ori，不是全表再现。未固定codec revisions和完整二进制/数据历史，3PTH结构核查不是数值验证。

已复用46适用全文与部分函数的同SHA阅读，并新增项目入口/准备脚本核查，128准备文本不冒称全部精读；3SVG和原输入Y平面实际看，无效果或视频主观证据。Git原件完整与平台外逐视频CSV缺口分开。本项同家族复用不计独立科研实证，不能解除MIE22未获得补充及历史/设备/时间未知。

后续价值在于明确伪teacher/亮度/原始RD与公共样本工具链的控制。实际CSV/曲线、数据许可/语义划分、模型及部署版本、手机完整成本与时序效用需后续合法新资料/授权实验解锁，当前只形成可证伪问题，不提前整体综合。
