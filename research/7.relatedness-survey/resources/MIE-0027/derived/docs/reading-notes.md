# MIE-0027 CBGAN：开放许可、可得摘要与正文获取缺口

2026-11-21；Codex  (sqs_A100) <langxibielangle@qq.com>。有效序30。实际0页PDF/0关键科学图表取得；不能称全文、公式或作者代码分析完成。

## 身份与具体相关性

出版方API coredata与Crossref确认Biomimetic Intelligence and Robotics，100369，DOI10.1016/j.birob.2026.100369，五作者Chaoyi Liu、Zhulin An、Chuanguang Yang、Libo Huang、Yongjun Xu。官方API写Available online31August2026及CC-BY4.0，ScienceDirect索引页面标InPress/JournalPre-proof。Crossref登记8月月份与8月27日license起点，不当实际首次公开日；未取得PDF，不能判确切排版版本/页数。

Crossref的51参考字段已逐项核读（可得结构化title/DOI及unstructured），b17完整书目与DOI准确对应MobileIE，ICCV2025 21949–21960，确认直接引用，正文语境/印刷[17]/实际骨干或loss继承仍unknown。b43出现FlowNet2不证明CBGAN使用流/时间模块，不能据摘要未提时间又断言无时间评价。

## 已取得的材料与真实访问过程

- 原2683B DOI返回其实是Redirecting HTML。只说明HTTP载荷完整；本项manifest标redirect_interstitial/usable_as_fulltext=false，不改旧回执。未执行其tracking/跳转JS，也不把它当摘要。
- 普通ScienceDirect和pdfft下载都403，均为832982B的错误HTML，SHA不同（请求诊断标识不同），属于同类错误页面；不是两份论文。API普通XML200、1966B仅coredata，无originalText/body；名字full-text-retrieval-response与openaccessType=Full均不是实际全文已得。
- Crossref19947B、OpenAlex24585B、Unpaywall2750B可核身份/书目/OA入口，后两只给DOI，没有直接PDF。公开CC-BY4许可已核，当前正文获取仍失败；不能把403/API401解释成论文未开放。
- 依Crossref声明的XML/plain链接与FULL view尝试时，同API host并发的三请求返回429RATE_LIMIT_EXCEEDED。停止并发；191.6秒后单次FULL view返回401AUTHENTICATION_ERROR InvalidAPIKey。PDF format返回406 restricted。未搜索/借用其他账户API key，没有持续循环重试。
- 自有Node/Playwright普通浏览器、默认automation身份、禁GPU。主页403并显示CPE00001内容未提供；截图已实际查看，非captcha图，未尝试解验证、伪装身份/代理或点击PDF。browser flag的access_challenge包括一般403，不表示实际看到了验证码。浏览器已关闭，0download。
- 作者组ChuanguangYang主页winycg.github.io正常取得70030B，只做当前论文可得性关键词/链接核查，bilateral/CBGAN/Chaoyi均零；不称精读全站其它研究。五组题名/DOI/模型/作者/排除出版站的查询快照与四个GitHub搜索保留。CBGAN三近名仓库不是可核作者实现；题名、user:winycg bilateral和ChaoyiLiu enhancement均零。没有发现可验证的本论文实现/明确公开代码承诺，不能推断私人或异题资料不存在。
- 共12新增普通GET、原1回执；普通浏览器另留HTML/PNG/SHA。所有正常终态，无在途下载/进程。没有购买、账户登录、外发联系、绕过访问控制。网站JS仅在普通浏览器运行，研究代码/模型从未执行。

## 可支持的作者主张

官方ScienceDirect索引摘要称Cycle Bilateral Generative Adversarial Networks（CBGAN）：低分辨率流在双边空间预测局部/全局仿射变换，再应用至全分辨率；使用无监督循环GAN、对抗、逆映射及cycle consistency。它报告4K照片几十毫秒，对常规ResNet生成器CycleGAN参数约少95.8%、速度超过170倍，以及无监督质量优势。以上仅为作者摘要归因，当前没有绝对参数/完整表或独立运行。

不能从这些词补写grid尺寸/guide函数/系数维度/卷积层数/归一化/插值边界/clip、两生成器和判别器的统计范围、训练配比和数据ID/许可、loss权重/seed、是否使用外部prior/高质量目标域。无配对监督不能直接称不需要目标域数据或完全不训练；inverse mapping是域间逆映射措辞，不是输出视频时间倒放或时间可逆。

“4K”“tens of milliseconds”没有精确shape/单一时间、设备/precision/backend/batch/warmup/计时边界/内存或热功耗；不能当手机实测/持续视频FPS。95.8%相对缩减也不能推出MobileIE四千参数级别；与一个重baseline的170倍不是相对MobileIE的速度。动机中的检测/分割/SLAM不等已做对应下游实验，摘要也未给这类结果。实际历史/未来/state/时间loss/指标未知。

## 具体价值、先例边界与后续控制

低分辨率预测、全分辨率施加变换可作为二期表示/成本参考，具体时序方案与有效性仍需正文。不能将双边grid/仿射/cycleGAN本身当本项目创新。书目中双边基础论文/HDRNet、检索浮出的2007同结构且HDvideo描述，以及FlowNet2都只作普通书目背景；未出现明确MobileIE接续/实现或其视频一致性具体路径，不因video关键词向全部前驱无限扩展。逐条处置见lead-dispositions.json；若新正文/代码出现具体合格关系，依只读Gate1增补再推进，不能忽略。

未来可证伪控制：实际unpaired目标域/数据重叠/训练预算与同条件基线；低分辨率grid/guide/fullres采样的完整计算/带宽/边界成本；真实运动/曝光/噪声下空间与时间/感知联合指标，分别考查guide/系数/输出的变化；完整端侧pipeline与相同precision/backend。都是全队列完成后筛选给Gate3，尚未执行或成立新方法。

受限支持身份、直接引用、摘要的机制大类和可审查获取记录；不支持全文分析、原数值领先、具体引用继承、手机/时间/新颖性判断。高优先补证保留：合法取得作者/正式正文、图表补充、确切项目/配置/权重及实验协议后，保留此终态另追加复审，不将受限统计成全文精读。
