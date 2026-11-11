# 5. 前期调研与重复性风险（Prior-Art Survey）

目标：在动模型之前，把 MobileIE 的引用图、同组接续工作、TeleAI/工业界线索、以及视频时序一致性方向梳理完整，回答三个问题：

1. 是否已经有人解决 MobileIE 的视频闪烁？（避免重复）
2. 导师提到的“电信接续研究”具体是什么？能否证实？
3. 我们的选题还剩下什么可发表的空间？哪些表述会被审稿人指为重复？

- 状态：**首轮完成，持续随实验增量核查**（最近2026-11-11）；计数快照仍标原日期，方法边界随新证据更新
- 关联：[4 号工作](../4.temporal-consistency/README.md)、[3 号基线](../3.video-flicker/README.md)、[2 号复现](../2.reproduction/README.md)
- 进展报告：[0008-T-261105-前期调研与重复性风险](../../docs/进展报告/0008-T-261105-前期调研与重复性风险.md)

## 1. 首轮核心结论（2026-11-05历史快照）

1. **未发现“MobileIE + 视频时序一致性/闪烁”的直接工作。** S2 抓到 33 篇引用、OpenAlex 18 篇、Crossref 15 次被引；逐条扫描标题与摘要，唯一明确的视频 LLVE 引用是 *AnyMod-LLVE*（ICML 2026，多模态重模型），不涉及极轻量单图网络扩展。
2. **同组接续工作有 5 篇，但全部是图像任务**：YUV 轻量低光（同第一作者）、LiteIE、IGDNet、频域双路径水下增强、DRNet 动态重参数化。它们是最近的“重复性风险源”，但没有一篇做视频/时序。
3. **TeleAI/中国电信线索未能证实。** 公开引用索引中没有 TeleAI 署名的 MobileIE 引用；TeleAI 2026 中文综述《极端环境图像退化、处理与应用研究综述》的 85 条公开参考文献中也没有 MobileIE。可能存在未被 Crossref/OpenAlex/S2 收录的中文期刊或内部工作，需要导师给出确切出处后复核。
4. **后处理是必须比较的强基线。** 首轮“降73%”来自旧实现；后续修复边界、warp方向与缓存问题后，当前结论应以4号工作报告0015及其前置报告为准。
5. 当时建议的差异化（尚未确认方法）：**JND/可见性加权的训练期时序一致性 + 重参数化兼容 + 4K 参数/手机实时**，并用逐模块消融解释“闪烁来自哪里”，避免被评价为“把已有 temporal loss 套到 MobileIE 上”。

## 2. 产出

| 产出 | 位置 | 内容 |
| --- | --- | --- |
| 检索方法与可复现性 | [`docs/01-search-method.md`](docs/01-search-method.md) | API、查询词、日期、计数口径、局限 |
| MobileIE 引用全表 | [`docs/02-mobileie-citations.md`](docs/02-mobileie-citations.md) | 33 条引用、分类、相关性 |
| 同组接续工作 | [`docs/03-direct-followups.md`](docs/03-direct-followups.md) | 5 篇同组工作 + 3 个竞赛报告的细节 |
| TeleAI/工业界核查 | [`docs/04-teleai-and-industry.md`](docs/04-teleai-and-industry.md) | 线索、证据、候选、验证方案 |
| 视频时序方法综述 | [`docs/05-video-temporal-survey.md`](docs/05-video-temporal-survey.md) | 对齐/循环/注意力/后处理/多模态/RAW 六类 |
| 轻量部署综述 | [`docs/06-lightweight-deployment.md`](docs/06-lightweight-deployment.md) | 重参数化、量化、蒸馏、延迟/能耗指标 |
| 新颖性与重复风险 | [`docs/07-novelty-and-overlap-risk.md`](docs/07-novelty-and-overlap-risk.md) | 空白、候选贡献、审稿人质疑与实验对策 |
| 后续受控探索相关工作 | [08](docs/08-exploration-literature-map.md)、[09](docs/09-feature-products-and-temporal-correlation.md)、[10](docs/10-streaming-denoising-and-equal-information-controls.md) | 训练、双线性/移位、流式去噪及等信息对照 |
| 数据与可靠性 | [11](docs/11-real-video-data-readiness.md)、[12](docs/12-motion-validity-and-quality-constrained-history.md) | ND实拍数据边界、运动有效性、质量约束 |
| 因果训练与多项式噪声 | [13](docs/13-training-objectives-and-causal-state.md)、[14](docs/14-polynomial-noise-and-bias-variance.md) | 近期训练先例、二次映射假设、偏差方差分析及未完成精读 |
| 状态及优化控制 | [15](docs/15-temporal-shift-baseline-scope.md)、[16](docs/16-optimization-and-reparameterization-controls.md) | 原TSM范围、可折叠分支与训练等价边界 |
| 已发表去噪复现入口 | [17](docs/17-published-denoiser-reproduction-readiness.md) | 官方权重/推理接入、未来等待与未完成LLVE对照 |
| 汇报材料 | [`report/prior-art-survey.html`](report/prior-art-survey.html) | 单文件 HTML，资源内嵌，可离线打开 |
| 原始数据 | [`data/`](data) | S2/OpenAlex 原始 JSON、Crossref 元数据、规范化 CSV |
| 抓取脚本 | [`scripts/fetch_citations.py`](scripts/fetch_citations.py) | 可重跑的引用抓取（带退避） |
| 规范化脚本 | [`scripts/normalize_citations.py`](scripts/normalize_citations.py) | 分类与 CSV 生成 |

## 3. 复现

```bash
PY=F:/dev-tools/python-envs/MobileIE-Neo/Scripts/python.exe

# 抓取（S2/OpenAlex 有速率限制，脚本会退避重试）
"$PY" research/5.prior-art-survey/scripts/fetch_citations.py

# 规范化
"$PY" research/5.prior-art-survey/scripts/normalize_citations.py
```

- [18 数据域与公开增强模型准备度](docs/18-data-domain-and-published-enhancement-readiness.md)：SDSD正文ND/ISO边界、StableLLVE真实推理、FastLLVE多尺度/时间/LUT先例。

- [19 配对证据与真实调光数据](docs/19-registration-evidence-and-real-lighting-data.md)：BVI版本/登录获取边界，R18实际OLiVES反证，避免用文献批评替代本项目测量。

## 4. 计数口径

| 来源 | 计数 | 说明 |
| --- | --- | --- |
| Semantic Scholar | 33 | 含 arXiv 预印本，作者/摘要最全 |
| OpenAlex | 18 | 以正式发表为主，含机构信息 |
| Crossref | 15 | 以 IEEE 正式版为准 |

三者不一致是正常的索引差异；正式论文中引用“截至 2026-11-05，MobileIE 在 S2/OpenAlex 分别有 33/18 篇引用”。

## 5. 局限

- Google Scholar 未做逐条人工核查（无法直接抓取），可能有遗漏；
- 中文期刊（知网/万方）覆盖不足，TeleAI 线索可能藏在这里；
- 引用列表是快照，2026 年新论文仍在增加；
- 摘要扫描只能发现显式提到 video/temporal 的工作，隐性使用无法仅靠摘要判断。

## 6. 下一步

- [ ] 让导师提供“电信接续研究”的确切论文/作者，或授权联系作者；
- [ ] 用 Google Scholar 人工复核 33 条引用之外的遗漏；
- [ ] 精读 YUV、LiteIE、DRNet、AnyMod-LLVE 四篇，确认是否含时序成分；
- [ ] 每月更新一次 `fetch_citations.py` 的输出，作为“查重台账”；
- [ ] 把 [`docs/07-novelty-and-overlap-risk.md`](docs/07-novelty-and-overlap-risk.md) 的差异化表述写进开题/论文 Introduction。

## 7. 修订记录

- 2026-11-05 · claude code (local)：创建 5 号工作，完成首轮引用抓取、分类与重复性风险分析。

- 2026-11-11T16:51:31.316795+08:00 · Codex  (sqs_A100)：补齐08–14增量入口，明确首轮计数/建议为历史快照，并移除旧光流改善数字的当前证据地位。

修改：2026-11-11T18:39:15.432022+08:00；作者：Codex  (sqs_A100)；原因：索引15/16，避免将训练优化或简化参考误作新颖性。

修改：2026-11-11T19:15:36.577012+08:00；作者：Codex  (sqs_A100)；原因：索引已运行FastDVDnet及BSVD入口核查。

修改：2026-11-11T20:24:45.731053+08:00；作者：Codex  (sqs_A100)；原因：索引R14/R15完整结果及新的结构/数据问题。

修改：2026-11-11T21:21:00.783613+08:00；作者：Codex  (sqs_A100)；原因：索引0027几何负结果与后续时间冗余问题。

- [20 历史利用与聚合训练先例](docs/20-reference-collapse-and-aggregation-training.md)：KPN参考帧局部极小值、辅助退火及短训练日程边界；R21无新增辅助损失。

修改：2026-11-11T22:23:36.759961+08:00；作者：Codex  (sqs_A100)；原因：索引已读原文/固定源码与重复风险核查。

- [21 噪声变换与warp假设](docs/21-noise-transforms-and-resampling-assumptions.md)：Noise2VST固定源码/正文阅读，空间插值矩与时间噪声方差的边界。

修改：2026-11-11T22:40:33.143257+08:00；作者：Codex  (sqs_A100)；原因：索引原模型统计适配的相关工作与自监督限制。

- [22 监督目标与条件假设](docs/22-training-objectives-and-conditional-assumptions.md)：Noise2Noise原理论/相关噪声段落阅读，R26梯度尺度与global clip耦合边界。

修改：2026-11-11T23:56:37.412147+08:00；作者：Codex  (sqs_A100)；原因：索引训练目标的前提与优化控制。

- [23 时间双线性先例](docs/23-temporal-bilinear-prior-and-operator-controls.md)：ABM原文/初始化先例及FST算子控制边界。

修改：2026-11-12T00:53:28.926200+08:00；作者：Codex  (sqs_A100)；原因：索引已核查先例，约束后续新颖性主张。

- [24 对应校准与独立性](docs/24-alignment-calibration-and-independence.md)：STBN原文/固定代码与精确选择反例，区分有监督和blind-spot假设。

- [25 骨干计算与流式状态](docs/25-compute-allocation-and-streaming-state.md)：R28路线冻结后的NAFNet/BSVD固定源码补读、TLC已知空间控制及未训练预算；乘法/移位/零初始化不作为新颖性，状态逻辑元素不替代真实storage/峰值。

修改：2026-11-12T01:32:47.707565+08:00；作者：Codex  (sqs_A100)；原因：索引对应与统计依赖的深度核查。

- [26 跨帧训练与推理边界](docs/26-cross-frame-training-and-inference-boundaries.md)：补读VLLVE++正文中的CFIM/训练期对应精化，区分参考帧依赖与单帧路径；近期方法的读取深度和未复现项分别保留。

修改：2026-11-12T05:16:58+08:00；作者：Codex  (sqs_A100)；原因：索引训练/推理依赖和同信息比较边界。
