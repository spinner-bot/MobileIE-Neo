# 检索方法与可复现性

- 创建：2026-11-05 · claude code (local)
- 检索日期：2026-11-05
- 目标论文：*MobileIE: An Extremely Lightweight and Effective ConvNet for Real-Time Image Enhancement on Mobile Devices*（ICCV 2025）
  - DOI：`10.1109/ICCV51701.2025.02038`
  - arXiv：`2507.01838`
  - OpenAlex：`W4417183307`

## 1. 数据源与接口

| 数据源 | 接口 | 用途 | 本次计数 |
| --- | --- | --- | --- |
| Crossref | `api.crossref.org/works/{doi}` | 正式发表元数据、被引计数 | 被引 15 |
| OpenAlex | `api.openalex.org/works/{doi}`、`?filter=cites:{id}` | 引用列表、作者机构 | 被引 18 |
| Semantic Scholar | `graph/v1/paper/DOI:{doi}/citations`、`paper/batch` | 引用列表、摘要、预印本 | 33 |
| 网页检索 | Web 搜索 | 补充中文/工业界/非索引工作 | — |

原始响应保存在 `data/`：

- `citations_s2.json`：33 条，含摘要（29/33 有摘要）；
- `citations_openalex.json`：18 条，含作者机构；
- `crossref_work.json` / `openalex_work.json`：论文元数据；
- `citations.csv`：人工分类后的规范化表。

## 2. 关键查询词

英文：

- `MobileIE lightweight image enhancement video temporal consistency flicker`
- `"MobileIE" cited video enhancement temporal consistency paper 2026`
- `MobileIE video flicker temporal`
- `lightweight video enhancement temporal consistency zero inference overhead reparameterization`
- `video low-light enhancement temporal consistency lightweight real-time 2025 2026`
- `flicker suppression video enhancement temporal consistency loss CNN mobile`
- `video deflickering real-time mobile low-light benchmark dataset 2026`
- `TeleAI low-light video enhancement temporal consistency MobileIE follow-up`
- `"MobileIE" "TeleAI"`、`"MobileIE" "China Telecom" citation`

中文：

- `"MobileIE" 视频 时序 一致性 增强 论文`
- `中国电信 人工智能研究院 低光视频增强 时序一致性 论文 2026`
- `"极端环境图像退化、处理与应用研究综述" MobileIE 参考文献`
- `TeleAI 低光 图像增强 视频增强 论文 2026 上海研究院`

## 3. 引用列表的判定流程

1. 拉取 S2/OpenAlex 全量引用；
2. 对标题 + 摘要做关键词扫描：`video`、`temporal`、`flicker`、`frame`、`sequence`、`recurrent`、`consistency`；
3. 逐条人工分类为：同组接续、视频/时序、轻量部署、竞赛、下游应用、无关；
4. 对疑似接续工作拉取作者列表，与 MobileIE 作者（Hailong Yan、Ao Li、Xiangtao Zhang、Zhe Liu、Zenglin Shi、Ce Zhu、Le Zhang）比对；
5. 对 TeleAI 线索单独核查（见 `04-teleai-and-industry.md`）。

## 4. 计数口径说明

- S2 的 33 条包含 arXiv 预印本与正式版重复项的可能性；本文按 S2 返回条目计数，未做 DOI 去重；
- OpenAlex 18 条、Crossref 15 条以正式发表为主，低于 S2 属于正常；
- 引用计数会随时间增长，本文数字仅代表 2026-11-05 快照；
- “同组接续”按作者重合判定，不代表机构/项目关系。

## 5. 局限

1. **Google Scholar 未逐条核查**：反爬与登录限制，可能遗漏未被 S2/OpenAlex 收录的引用；
2. **中文文献覆盖不足**：知网/万方/维普未系统抓取，TeleAI 等机构的中文期刊工作可能缺席；
3. **摘要缺失**：33 条中 4 条无摘要（#5、#11、#15、#16），只能按标题判断；
4. **隐性相关工作**：摘要不提 video/temporal、但方法可用于视频的工作无法自动发现，需要后续精读；
5. **工业界内部工作**：未公开发表的工作无法通过学术索引发现，只能依赖导师/作者提供线索。

## 6. 复现方式

```bash
PY=F:/dev-tools/python-envs/MobileIE-Neo/Scripts/python.exe
"$PY" research/5.prior-art-survey/scripts/fetch_citations.py
"$PY" research/5.prior-art-survey/scripts/normalize_citations.py
```

脚本会对 429/503 做指数退避；若持续失败，可降低请求频率或配置 API key（OpenAlex 免费 key 可提高额度）。
