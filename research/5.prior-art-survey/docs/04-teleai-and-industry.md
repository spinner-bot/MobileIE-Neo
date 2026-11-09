# TeleAI / 中国电信“接续研究”线索核查

- 创建：2026-11-05 · claude code (local)
- 来源：导师会议转写（Standard recording 68）：“电信的 AI 部门……上海的研究院……有老师用了他这个往后做了一点……他是看到他的那个引用的那个。”
- 目标：确认是否存在“TeleAI（中国电信人工智能研究院，上海）在 MobileIE 之后做的视频/图像增强接续工作”。

## 1. 结论先行

**截至 2026-11-05，未找到可公开证实的 TeleAI 署名、且引用 MobileIE 的论文。**

具体核查：

| 核查项 | 结果 |
| --- | --- |
| Semantic Scholar 33 条引用 | 作者列表中无李学龙、陈翔宇（Xiangyu Chen）等 TeleAI 常见作者 |
| OpenAlex 18 条引用（含机构） | 无 TeleAI / China Telecom 机构 |
| Crossref 15 次被引 | 未发现 TeleAI |
| Web 检索 `"MobileIE" "TeleAI"` / `"MobileIE" "China Telecom"` | 无关联结果 |
| TeleAI 2026 中文综述《极端环境图像退化、处理与应用研究综述》 | 抓取到 85 条公开参考文献，**未检索到 MobileIE** |
| TeleAI 低光增强 ReDDiT（CVPR 2025） | 论文 HTML 中未检索到 MobileIE |
| TeleAI 通用低层视觉 Lumina-OmniLV（arXiv 2504.04903） | 论文 HTML 中未检索到 MobileIE |
| TeleAI 水下生成式复原（arXiv 2511.14521） | 论文 HTML 中未检索到 MobileIE |

## 2. 已核实的 TeleAI 相关线索

| 工作/线索 | 与 MobileIE 的关系 | 说明 |
| --- | --- | --- |
| 《极端环境图像退化、处理与应用研究综述》（赵致远、高君宇，TeleAI 上海，2026） | **无引用证据** | 覆盖极端弱光、雾霾、水下；85 条公开参考文献中无 MobileIE；可能 PDF/正文有未渲染条目，需下载全文复核 |
| ReDDiT：*Efficient Diffusion as Low Light Enhancer*（CVPR 2025，李学龙参与） | 无引用证据 | 低光图像增强，10 个基准含 SDSD；是图像方法，不是视频时序 |
| Lumina-OmniLV（arXiv 2504.04903，Xiangyu Chen 为 TeleAI） | 无引用证据 | 通用低层视觉多模态框架；可能作为“TeleAI 低层视觉能力”的代表 |
| TeleAI 水下生成式复原（arXiv 2511.14521） | 无引用证据 | 图像复原/增强，非视频 |
| TeleAI GVC / 正激励摄像头 | 无引用证据 | **视频压缩/通信**方向，不是图像增强；导师可能把两条线混在一起 |

## 3. 可能的解释

1. **导师看到的是未被 Crossref/OpenAlex/S2 收录的条目**：例如中文期刊、会议论文、内部技术报告，或引用格式不规范导致解析失败；
2. **“引用”可能来自作者本人的汇报/主页**，而非正式论文；
3. **导师记忆中的“电信”可能指其他机构**（如某手机厂商 AI 部门），或把 TeleAI 的视频压缩工作与 MobileIE 混淆；
4. **确有内部工作但未公开**：这种情况下无法通过学术索引发现。

## 4. 验证方案（按成本从低到高）

1. **请导师给出确切出处**：论文标题、作者、会议/期刊、年份，或截图；这是最快路径；
2. **Google Scholar 手动查看 MobileIE 的“被引用”列表**，逐条核对机构（需要浏览器/账号）；
3. **IEEE Xplore 的 Cited By**：MobileIE DOI 页面（[IEEE 11444461](https://ieeexplore.ieee.org/document/11444461)）可能列出正式引用；
4. **知网/万方/维普**：检索“MobileIE”“移动端 图像增强 实时”“电信 人工智能研究院”等中文关键词；
5. **TeleAI 官网/公众号/arXiv 作者页**：关注 Xiangyu Chen、李学龙、赵致远、高君宇等；也可查 TeleAI 招聘/成果页；
6. **直接联系作者**：导师认识 MobileIE 作者，可在项目申报结果出来后询问“是否有 TeleAI 团队引用/扩展”。

## 5. 对我们选题的影响

- 如果 TeleAI 的接续工作只是**引用 MobileIE 做图像增强**（如综述、ISP、水下），不影响我们的视频时序选题；
- 如果确有**视频/时序**接续工作，需要立刻获取全文，重新评估新颖性；
- 在未证实前，论文 Introduction 不应写“据我们所知，TeleAI 尚未……”这类无法证实的表述；只写“公开文献中未发现 MobileIE 的视频时序扩展”。

## 6. 建议的记录方式

在开题/周报中保留以下表述：

> 截至 2026-11-05，我们在 Semantic Scholar（33 篇）、OpenAlex（18 篇）、Crossref（15 次）的 MobileIE 引用列表中，未发现针对视频时序一致性/闪烁的工作；导师提到的 TeleAI 接续研究尚未获得可核实的出处，待补充。

## 7. 待办

- [ ] 向导师索取确切论文信息；
- [ ] 用 Google Scholar / IEEE Cited By 人工复核；
- [ ] 知网/万方检索 TeleAI 中文文献；
- [ ] 若找到 TeleAI 论文，精读并更新 `07-novelty-and-overlap-risk.md`。
