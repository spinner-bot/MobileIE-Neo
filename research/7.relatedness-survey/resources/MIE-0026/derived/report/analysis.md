# MIE-0026｜REZS-Edge：直接引用已核，正文与实现证据受限

2026-11-21；Codex  (sqs_A100) <langxibielangle@qq.com>。有效序29，拟按closed_limited结项，绝非全文或代码分析完成。

此项值得后续优先补证：出版方登记确认其直接引用MobileIE，摘要涉及噪声/照明与结构重参数化的组合，可能构成二期相关先例。当前只有官方摘要与元数据，不能判定具体继承关系、方法重复程度或实验是否成立。

## 已确认的证据

[IEEE正式页面](https://ieeexplore.ieee.org/abstract/document/11585811)给出八作者、IEEE Internet of Things Journal 13(17)，39626–39640，在线2026-06-29，期次2026-09-01。[出版方Crossref登记](https://api.crossref.org/works/10.1109%2FJIOT.2026.3707835)中reference key ref60的DOI精确指向原MobileIE；这确认引用关系，尚不能证明正文语境或印刷序号。见[身份](../../identity.json)、[引用溯源](../docs/citation-provenance.json)。

摘要把REZS-Edge描述为无配对监督的照明/噪声增强框架，结合Retinex、noise-to-noise与全局/局部照明约束，并采用结构重参数化；列LOLv1/LOLv2-Real、Snapdragon8Gen3的6.25ms及若干下游感知应用。这些仅为作者报告。缺少正文公式、表和协议，无法核查其SOTA、泛化或部署完整性。

## 获取范围和明确缺口

新增十次正常GET及初始两回执均已终态；取得官方摘要HTML与三学术元数据、零结果arxiv响应，共五份有内容原件。stamp、getPDF、官方声明的iel8入口均202空；reference/figure入口200空。旧采集的空响应success只在本项manifest更正，原回执保留。当前官方访客标志未提供全文访问，OpenAlex/Unpaywall无OA位置；不能用空响应推断某种特定封禁原因。

标题/模型/作者检索和第一作者名对应GitHub线索未定位确切REZS实现。EMOD是2025另一篇工作，不作为本项代码；宽泛搜索中的list与异名项也不证明实现存在或不存在。没有登录、购买、联系作者或绕过访问控制。所有来源/失败见[清单](../../sources/manifest.json)、[可得性审查](../docs/availability-review.json)、[实际阅读笔记](../docs/reading-notes.md)。

完整正文预期15页，实际0页PDF、0张图表取得。训练/测试适配成本、噪声假设与配对、照明约束、重参数化分支/折叠条件、参数/MAC/峰值、数据ID/许可/外部先验、原表/消融、手机输入/precision/delegate/计时边界及文件SHA均未知。历史/未来帧、state、因果性和连续视频评价也未知，不能从应用场景补写方法。

## 对二期研究的可用价值

摘要6.25ms的倒数为160/s，但不是已证持续视频吞吐。即便同芯片，也不能直接与MobileIE的不同输入/精度/后端/计时范围数字排序。零样本也不自动表示零训练或零测试适配成本。

后续需回答四个可证伪问题：zero-shot实际优化哪些信息且费用多少；照明/噪声配对假设能否适应真实相关噪声与曝光；可折叠结构与MobileIE具体有什么关系及数值等价边界；完整端侧成本和空间质量—时间/感知能否同时改善。当前不具备回答它们的材料，不将其写成结论或创新成立。

合法取得正式或作者正文、确切官方实现/配置/权重/协议后，保留本轮终态并追加复审。本项受限缺口必须进入总体报告，不能并入“全部论文全文精读”。Gate3只交接候选，等待下一实验分支任务。

主张见[12项claims](claims.json)，结项见[Gate2审查](../docs/gate2-review.md)及[决定](gate2.json)。0模型加载/推理/训练/GPU/Gate3。
