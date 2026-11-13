# 公开配对视频：RAW/sRGB接口与来源身份边界

核查：2026-11-12T08:15:48+08:00；作者：Codex  (sqs_A100)。R40完整CPU评价运行期间，针对真实域缺口做一次有界来源核查，没有取得新像素或变更最终保留策略。

[DarkVision原文v1](https://arxiv.org/html/2301.06269)阅读§3.2–3.3及§5.4：提供12-bit TIFF和RAW；同一透过率的亮暗对可配对，不同照度档是依次采集，不能跨档按帧号拼GT。其手机示例先去黑电平、demosaic，再按指定子集强度尺度处理。因此RAW/已渲染sRGB与输入标定要明确，不能从“有手机示例”推断任意ISP视频可直接输入同一8bit增强器。这也不是证明R38失真的具体根因。

[含2025 DOI编号的出版页](https://www.sciopen.com/article/10.26599/CVM.2025.9450461)在本次工具中仅呈现动态页面壳；未取得完整更新稿或权威数据下载入口，不把2023预印本中的未来发布承诺当当前可下载事实。

检索命中[同名HF仓库](https://huggingface.co/datasets/DarkVision/dataset)，连接器详情只确认该仓库存在。公开viewer报错出现`image_labels_fake.csv`/`fake_image`，与论文描述对应关系尚未确认；没有作者交叉链接/数据说明，**不将它认定为论文官方低光配对数据**，不下载或预览其像素。HF连接器dataset_search被服务端禁用，hub_repo_details可用；随后使用公开作者/出版页面核查，未绕过登录或借用服务器其他用户数据。

[SDSD作者仓库](https://github.com/JIA-Lab-research/SDSD)当前README仍明确indoor70/outdoor80视频对及机电复现动态的采集设置。本轮未重新请求已知Drive大文件或更改既有访问限制；可用作者说明不等于像素已取得。DID/BVI的此前访问记录也不自动变成新完成资料。

接续：先完成已有合成/真实配对资料的固定R40比较；若需要新增完整基准，先确认作者来源、发布版本、划分与渲染/标定流程，再登记读取范围。检索命中不代表形成有效新基准，不为凑主表打开最终保留像素。

## 出版元数据增量核对（2026-11-12T13:10:09.132243+08:00）

同一[出版页](https://www.sciopen.com/article/10.26599/CVM.2025.9450461)本次可读取摘要及文献元数据：正式卷期为2026年12(3)，615–642页，网页列Published 27 March 2026。DOI里的2025不能当作正式出版年份。前述动态页面壳是08:15该次读取事实，现已补充可读取范围；本轮仍未取得完整更新PDF或权威数据下载入口，也未取得新像素。旧arXiv v1接口结论仍只归属于该版本，不能自动覆盖正式全文。

## 新来源3L-D的实际准备度（2026-11-12T15:16:29+08:00）

作者：Codex  (sqs_A100)；原因：R44最终评价等待期，只针对外部真实低光验证缺口核查一个新来源，不增候选网格。

[3L-D v1](https://arxiv.org/html/2611.02695v1)定向阅读§2.2–2.4及§3：受控缩比场景/重复机械轨迹/多档夜间照明，不当作自由动态自然夜景。论文交叉链接到作者站；网站固定提交2013e3d570c8903d5cdaff586b202f7e59618cc5的index.html进一步链接[官方HF样本](https://huggingface.co/datasets/3L-OPTRONICS/3L-Dataset)。README旧账户标签与当前下载链接不同，保留差异；不采用仅同名的任意镜像。

HF连接器实测为gated、imagefolder、少于1K条样本；说明要求先申请。未申请或发送信息，未下载/解码像素。连续视频布局、RGB适用性及正式划分尚未验证，不能据论文提到视频就将公开图像样本计为视频基准。[元数据记录](../../4.temporal-consistency/data/external-3ld/source-readiness.json)独立提交。本次没有新完成夜间验证，也不绕过权限或反复请求已受限归档；继续等待已启动R44完整评价。

## RAW视频备选的有界补核（2026-11-13）

作者：Codex  (sqs_A100)。R73原训练等待期，仅核官方资料，不扩大模型网格。[实际来源记录](../../4.temporal-consistency/data/external-raw-video-readiness/source-reading001.json)含每个README/下载脚本blob身份和实际阅读范围，0新像素/CNN/SGD/GPU；不重试既有SDSD配额或其他需授权入口。

[CRVD作者仓库](https://github.com/cao-cong/RViDeNet)已有官方数据入口，注明GBRG、黑电平240、白电平4095，并保留原始平均参考与BM3D处理的区别；训练scene1–6与测试7–11亦有说明。它可作为后续真实RAW噪声资源，但尚未取得本地文件或固定RGB渲染，不能直接计入当前RGB LLVE表，更不能据此宣称已补齐自然夜间配对。这里只读仓库资料，未声称复现原方法或完整读完论文。

[SMOID原文](https://openaccess.thecvf.com/content_ICCV_2019/papers/Jiang_Learning_to_See_Moving_Objects_in_the_Dark_ICCV_2019_paper.pdf)§3.1–3.2和§4.1描述在足够照明下以1%透过率ND和双相机采集亮暗对；这是实测衰减观察，不等于全为自然暗夜。12-bit值存16-bit容器、RAW Bayer及裁剪奇偶性影响渲染，原方法还作片段强度缩放，不能直接替代既有RGB接口。[作者README](https://github.com/MichaelHYJiang/Learning-to-See-Moving-Objects-in-the-Dark)只给示例入口；读取下载脚本也只确认三组标明example的ZIP，未执行或请求其数据主机，不把它计为完整179对及公开划分已取得。

上述核查只完善准备度边界，未补齐自然夜间/独立最终证据。若后续确需RAW噪声验证，先单独登记来源、固定训练/开发范围、原始参考类型及公共渲染，不把元数据变成下载完成，也不为新增一张表绕开当前候选的质量门槛。

## CRVD原生训练输入已取得（2026-11-13 09:03 CST）

后续按[独立来源资格协议](../../4.temporal-consistency/docs/crvd-source-qualification-protocol.md)推进，现已实际取得[scene1/ISO1600/frame1固定四TIFF](../../4.temporal-consistency/data/external-crvd/README.md)，不再将该来源当前状态笼统描述为“只有元数据”。其中两个噪声重复和原始平均为uint16，BM3D参考为float32；ZIP CRC/大小及原生TIFF标签/读出类型核验正常，未渲染/转换或评价模型。仅一个训练时刻，不是完整视频、传感器标定、自然暗夜或独立最终；测试像素仍未读。官方包实际按场景ZIP发布，初始路径记录错误及按注册结构停止均保留，完整过程/有限Range传输见[日志](../../4.temporal-consistency/process/external-crvd/journal.md)。原文主§3.1和补充§1.1/表1的人工运动/多帧平均假设已进一步核查，不直接移入现有RGB方法主表。

## DarkVision获取状态接续（2026-11-13 11:40 CST）

[定向补读56](56-darkvision-access-and-temporal-baseline-scope.md)核v1末尾Data Availability的通讯作者合理申请声明，并补TC-DVE §4.1/评价§5.2；当前出版/作者页仍未取得可确认的官方数据入口，未核同名HF不作来源。未下载像素、联系作者或复现模型；这不是证明不存在公开版本，暂不循环重复查找，先完成原R74。
