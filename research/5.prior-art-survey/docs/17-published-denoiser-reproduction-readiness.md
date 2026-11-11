# 17｜已发表去噪参考的实际接入

创建：2026-11-11T19:15:36.577012+08:00；作者：Codex  (sqs_A100)；原因：R12运行FastDVDnet，明确原始权重、输入假设和未来等待；承接调研10。

## FastDVDnet：已完成固定权重推理接入

[官方仓库固定commit](https://github.com/m-tassano/fastdvdnet/tree/c8fdf6182a0340e89dd18f5df25b47337cbede6f)的README、models.py、fastdvdnet.py、test_fastdvdnet.py和MIT许可本次已读；所需模型/推理源码原字节vendor且Git blob SHA一致。README说明训练噪声范围5–55，以及原共享权重基于旧DALI/PyTorch训练，更新训练代码并不保证复现同表现。本项目只用已有权重，未训练原模型。

原推理以五帧为中心输出，前后各两帧；边界reflect，H/W pad到4倍数后去除。原test脚本会额外添加AWGN，不能直接套到已有退化；本项目调用原网络和temp_denoise，不加噪声。noise map使用sigma/255，输入/输出RGB归一化，官方clip权重与source/bytes/Git对象/SHA256已核验。

独立固定smoke与官方完整推理最大差0，clipped AWGN25 PSNR+9.47dB；只能证明加载、归一化和路由正确。R12低光迁移另有920行，其成功/失败不当作论文DAVIS标准结果。five_current重复同帧不是独立发表的单帧网络，而是同网络无真实时间信息对照。

两未来帧等待与计算时间分别报告。2.479M参数的去噪器有原生数据窗口，并不意味着4075参数MobileIE能直接获得同样收益，也不能把A100阶段耗时当手机端到端速度。

## BSVD：仅本轮入口及README核查

[原论文作者/源码链接](https://arxiv.org/abs/2207.06937)与[官方README](https://github.com/ChenyangQiQi/BSVD/blob/main/README.md)已核查，README blob bc91eedd129a9cf6d5a0a4e79a54a809f68f2134。RGB synthetic分支基于BasicSR/DALI，权重列OneDrive入口，另有blind/unidirection权重和bsvd_raw分支。本轮未下载权重、未安装旧环境，也未据此认定下载不可用。raw分支不能直接当RGB低光参考。

调研10已读论文§3.1–3.4并记录y(i−N)输出延迟；本轮不重复声称精读或将streaming等同零未来。后续若接入须核查具体配置的N、flush与输出索引，并与MobileIE同延迟预算比较。

## 研究约束

已发表通用去噪、低光增强模型和本项目小参数候选仍是不同参考类别；不能用FastDVDnet迁移替代正式LLVE模型复现。R12真实多帧增量必须对照five_current，并在native/resize分别说明，保留全部sigma而不事后挑最有利域。更复杂模型有效，首先提示数据/对应/训练与容量差距，不能直接推出新模块创新。

关联：[0022报告](../../../docs/进展报告/0022-T-26111119-已发表去噪对照与原生窗口.md)、[源清单](../../../third_party/fastdvdnet/UPSTREAM.json)、[权重manifest](../../4.temporal-consistency/data/exploration-r12-fastdvdnet/manifest.json)。
