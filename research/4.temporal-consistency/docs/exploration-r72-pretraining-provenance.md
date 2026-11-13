# R72参考的公开起点与项目适配来源

2026-11-13；作者：Codex  (sqs_A100)。补充R72原有预训练限制，不重写已结束结果或原协议。依据[固定版本来源阅读](../process/exploration-r72/pretraining-provenance001/source-reading.json)与[十个最终权重谱系](../process/exploration-r72/pretraining-provenance001/lineage.json)，0新增像素/CNN/SGD/GPU。

## 已核信息

|R72参考|实际公开起点与作者声明|项目内适配谱系|仍未证明的事项|
|---|---|---|---|
|MobileIE MSE|公开lolv1_best_slim；原配置指向LOL-v1 our485，论文§4.1列LOL任务数据|同seed R41 expanded4000 → R44 square_mse新4000|公开二进制完整逐样本训练史、跨数据集内容去重|
|MobileIE L1|同一公开LOL-v1起点|R69从原expanded初始化独立训练8000|不能将其与MSE的优化器/训练阶段差异抹掉|
|Nano|固定仓库README把所用Nano权重对应LOL-v1；已存官方选项亦指向LOL-v1|R56从公开权重适配8000|权重完整训练日志和额外预训练排查|
|StableLLVE L1|固定README标该权重为synthetic-clean；论文§4.2描述DAVIS源合成资料|R65从公开clean权重独立作spatial_l1适配8000|论文随机87视频训练划分与公开二进制逐源一一对应|
|FastDVDnet|固定README声明DAVIS2017训练；既存作者清单有90个ID，使用clipped-AWGN公开权重|R53从原公开权重作source-context适配8000|作者声明不等于二进制全训练日志或跨数据集像素去重|

MobileIE原配置：[7030d12/config/lle.yaml](https://github.com/AVC2-UESTC/MobileIE/blob/7030d12/config/lle.yaml)；论文[§4.1](https://arxiv.org/html/2507.01838v1)。两个MobileIE最终权重并非从随机初始化开始，不能把项目8000次更新当作总预训练预算。

Nano依据同一[固定README](https://github.com/albrateanu/multinex/blob/e15790a963935569fff1ba6a8b0d33a791765749/README.md)的资料和权重对应，以及仓库已保存的`Options/Multinex-Nano_LOL-v1.yaml`，不是按论文总表猜测本项目用的是哪一个检查点。

StableLLVE的[固定README](https://github.com/zkawfanx/StableLLVE/blob/4cc070da328ffadc8498285985aa98175e473bc5/README.md)仍保留clean权重不适配噪声的提示。[原文§4.2](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Learning_Temporal_Consistency_for_Low_Light_Video_Enhancement_From_Single_CVPR_2021_paper.pdf)从DAVIS各划分筛出107个1080p视频，再随机分87/20；这不是直接采用本项目DAVIS的train/DEV划分。没有取得与该二进制对应的完整87-ID清单，不将论文来源声明升级为逐ID已排除重叠。原PDF新请求403，索引缓存原文可读，访问范围已记录。

FastDVDnet的[固定README](https://github.com/m-tassano/fastdvdnet/blob/c8fdf6182a0340e89dd18f5df25b47337cbede6f/README.md)及[已有90-ID核对](../data/external-davis/fastdvdnet-training-overlap.json)确认，旧bear/lucia/walking开发源确实在其作者训练清单中。该事实不因新增REDS而消失，也不能把旧DAVIS比较重新叫未见泛化。原共享权重训练时的软件版本与新训练入口不同，继续保留原复现限制，不新增DALI环境或复现网格。

十个实际最终适配配置均记录相同57个DAVIS训练ID与12个OLiVES白天ND源，manifest摘要一致；源身份不包含REDS。FastDVD额外源上下文、各模型容量/公开预训练/目标/优化器及训练计算仍不同，不能据相同ID或8000更新声称严格单因素架构对比。

## 采用的论文措辞

可以说“REDS来源未用于本项目适配，公开起点的已查作者声明指向LOL或DAVIS”；应同时说明未掌握所有公开权重的逐样本完整训练史，也未作跨数据集内容指纹排查。这比笼统未知更具体，但不是对任意未披露预训练的排除证明。

R72四源已用于开发，且退化为合成低光；即使将来排除全部预训练重叠，也不会因此成为自然夜间、独立最终测试或严格等训练预算SOTA表。不改R72旧`pretraining_REDS_overlap`字段及原始来源文件，只在此追加证据层级与边界。R73原冻结训练继续，其最终门槛不回写。
