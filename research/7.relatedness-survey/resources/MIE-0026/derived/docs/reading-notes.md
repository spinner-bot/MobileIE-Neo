# MIE-0026 REZS-Edge：可得证据与正文缺口

2026-11-21；Codex  (sqs_A100) <langxibielangle@qq.com>。有效序29。本文只能按证据受限口径审查，不能称全文或代码分析完成。

## 身份与直接相关性

IEEE Internet of Things Journal 13(17)，39626–39640，期次2026-09-01，在线/插入日期2026-06-29。IEEE公开HTML的标题/DOI/八作者/出版信息与Crossref一致，后者以期次月份登记，不当日期矛盾。页码范围预期15页，实际0页PDF取得、0张关键图表查看。

正式出版方在Crossref登记的67条reference中，key ref60的DOI为10.1109/ICCV51701.2025.02038，与原MobileIE准确匹配。这个证据比仅逆引用索引强，但不能推出正文引用位置、实际采用哪个结构/loss、比较列或是否批评MobileIE，亦不把key当已读印刷[60]。67条登记记录/DOI列表已核，其他普通参考DOI不自动扩成新MobileIE资源。

## 实际可得材料与读取范围

- 官方IEEE摘要HTML58150B正常200、实际JSON字面解析，不执行页面JS。摘要、作者名/机构、出版与访问标志、关键词和入口已读。作者生平/下载计数不是技术证据，不据索引词推断结构；例如自动Index Terms不能代替方法正文。
- Crossref11426B、OpenAlex22895B、Unpaywall1958B元数据留存，核身份、登记引用及OA位置；不是三个独立实验或三篇全文。arxiv精确标题API771B、totalResults=0，不等没有异题预印本。
- 两轮检索响应保留：精确标题/DOI/模型名/作者及PDF/arxiv/GitHub线索，web打开IEEE有418/不可访问，普通采集稍后200摘要，两种工具响应不同如实留存。
- GitHub模型查询一异名项、精确标题零项、宽泛作者词第一页20项，未定位可核为本论文官方实现的直接项。第一作者名对应线索XiongZhongxia只查到EMOD，实际读取其metadata/README，是2025另一篇多光谱检测论文，没有REZS链接；不把同名/普通list/另一篇实现当REZS。未对无关代码开展下一资源分析，不宣称穷尽私人托管或全部网络。

## 获取尝试终态

初始两个202空响应曾被旧采集程序记success，本项manifest纠正为empty_response/complete=false，旧回执保留。新增十个正常GET：IEEE摘要200有内容；stamp、getPDF、官方metadata精确声明的iel8 PDF均202空；公开reference/figure REST均200空；三学术metadata200；arxiv API200零条。12回执（含初始二）均有时间/URL/真实响应，五个有原始内容存档及SHA；无在途下载。

IEEE当前访客字段isOpenAccess=false、isFreeDocument=false、openAccessFlag=F、fullTextAccess=false；OpenAlex/Unpaywall未给OA位置。标记figures/references=true只表示导航存在，正常接口未取得内容，不能当图表已看。HTTP空响应不被解释成特定封禁/付费原因；没有登录、付费、外发联系、验证码绕过或使用Crossref中用途为similarity-checking的staging入口。只有公开声明的正常PDF入口GET。

## 能读出的作者主张与不能补写的内容

摘要将方法称REZS-Edge，描述Retinex、noise-to-noise、全局/局部照明约束和结构重参数化；列出LOLv1/LOLv2-Real、Snapdragon8Gen3上6.25ms以及检测/实例分割/姿态应用。均只归于作者摘要，不复述为本次独立确认。

没有正文，不能补出噪声配对/独立性与均值假设、illumination参数化、重参数化具体分支/折叠条件、参数/MAC/峰值、输入范围、监督及外部先验、训练数据ID/许可、测试时优化次数、原图表/指标/seed/消融。zero-shot可能对应不同训练/测试设定，不能径直声称逐图训练或完全不训练。未知视频历史/未来/state/因果性，不能从自动驾驶与机器人应用推断连续视频实验。

6.25ms倒数为160/s，只是摘要数值的条件倒数；分辨率/precision/delegate/计时边界/初始化与适配成本/手机文件SHA/热功耗未得。不能与MIE1同芯片的6.72ms直接排序，不能声称持续160FPS或空间/时间联合优势。SOTA与下游泛化缺具体表、数据划分与对照，不能验证。

## 科研价值与结项边界

资源仍值得高优先补正文：它直接引用MobileIE，且摘要已覆盖噪声、照明和可折叠部署这一相关组合，二期不能无视该先例风险；但未知不能变成肯定采用MobileIE骨干/精确重复或创新已成立。

下一阶段的可证伪控制是：明确zero-shot训练与测试适配费用；同数据/分辨率/precision/后端比较完整成本；核噪声假设与真实相关噪声/曝光；核折叠数值等价和真实运动下空间质量—时间/感知联合表现。只是待全队列完成后的Gate3候选，当前0模型/推理/训练/GPU/Gate3。

受限结项支持官方身份、确切登记引用、摘要归因与缺口/控制，不能支持全文/实现/质量或速度复现。解锁条件为合法取得正式/作者正文或确切官方项目、配置/权重/协议；保留本轮所有材料和终态，再追加复审，不回写旧闭项或把证据受限统计成全文精读。
