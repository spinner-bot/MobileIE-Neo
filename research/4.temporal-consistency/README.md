# 4. 时序一致性方法（Temporal Consistency）

目标：以论文为最终产出，深入核查 MobileIE 的架构与训练机制，在移动端计算预算内设计视频增强模型，用传统方法、后处理和已有视频模型对照验证画质、时间稳定性与效率收益。

- 状态：**前期探索，多轮受控预实验/机制诊断及必要跟进已完成**（2026-11-11）；没有确定最终方法，已建立实拍ND衰减数据与两域强控制，完整训练及自然夜景泛化尚待验证
- 当前研究入口：[MobileIE 架构、训练核查与视频研究设计](docs/architecture-and-video-research-plan.md)
- 最新结果：[进展报告0029](../../docs/进展报告/0029-T-26111122-动态聚合与增强对齐顺序强控制.md)、[同流场全表](report/exploration-r20/tables.md)、[图](report/exploration-r20/dynamic_order_tradeoffs.png)、[过程](process/exploration-r20/journal.md)，1472行/368原基线回放差0，固定前置未成独立动态优势
- 进行中：[R21聚合感知训练协议](docs/exploration-r21-protocol.md)，同R14 native任务，四类×双seed×4000=32000更新；独立模型/Adam只共享固定history准备，不称新方法
- 已知信息诊断：[进展报告0028](../../docs/进展报告/0028-T-26111121-可用时间信息与非线性聚合顺序.md)、[信息图](report/exploration-r19/known_temporal_information.png)、[顺序图](report/exploration-r19-aggregation/aggregation_order.png)、[过程](process/exploration-r19/journal.md)
- 配对反证：[进展报告0027](../../docs/进展报告/0027-T-26111121-监督配对几何反证与数据准备.md)、[图](report/exploration-r18/pair_geometry.png)、[过程](process/exploration-r18/journal.md)，104项未支持大整体整数平移主因
- 第十六轮：[进展报告0026](../../docs/进展报告/0026-T-26111121-多尺度骨干与独立历史反证.md)、[完整表](report/exploration-r16/tables.md)、[历史反证图](report/exploration-r16/multiscale_history_controls.png)、[过程](process/exploration-r16/journal.md)，24000更新/3680新逻辑行完成
- 数据准备：[R17协议](docs/exploration-r17-protocol.md)已登记，SDSD下载配额限制；BVI-RLV公开元数据可读、像素需IEEE登录，均尚无新像素结果。下一步检验时间冗余与噪声相关性
- 第十五轮：[进展报告0025](../../docs/进展报告/0025-T-26111120-公开增强权重迁移与数据域核查.md)、[公开clean增强迁移表](report/exploration-r15/tables.md)、[过程](process/exploration-r15/journal.md)
- 第十四轮：[进展报告0024](../../docs/进展报告/0024-T-26111120-原生训练与深度交互对照.md)、[深度/尺度全表](report/exploration-r14/tables.md)、[图](report/exploration-r14/native_depth_controls.png)、[过程](process/exploration-r14/journal.md)，42000新更新及全部Adam/checkpoint审核完成
- 第十三轮：[进展报告0023](../../docs/进展报告/0023-T-26111119-前置历史与训练状态分布反证.md)、[表](report/exploration-r13/tables.md)、[过程](process/exploration-r13/journal.md)
- 第十二轮：[进展报告0022](../../docs/进展报告/0022-T-26111119-已发表去噪对照与原生窗口.md)、[完整表](report/exploration-r12/tables.md)、[图](report/exploration-r12/published_denoiser_transfer.png)、[过程](process/exploration-r12/journal.md)
- 第十一轮：[进展报告0021](../../docs/进展报告/0021-T-26111119-强优化与历史对应反证.md)、[同权重历史图](report/exploration-r11/strong_history_controls.png)、[表](report/exploration-r11/tables.md)
- 第十轮：[进展报告0020](../../docs/进展报告/0020-T-26111118-原生尺度与匹配区域诊断.md)、[平滑权衡](report/exploration-r10/matched_smoothing.png)
- 第九轮：[进展报告0019](../../docs/进展报告/0019-T-26111118-优化基线与新增开发集.md)、[完整表](report/exploration-r9/tables.md)、[优化图](report/exploration-r9/optimizer_controls.png)、[过程](process/exploration-r9/journal.md)
- 第八轮：[进展报告0018](../../docs/进展报告/0018-T-26111118-输入时域化与优化强对照.md)、[完整表](report/exploration-r8/tables.md)、[优化/历史图](report/exploration-r8/optimization_history.png)、[过程](process/exploration-r8/journal.md)
- 第七轮：[进展报告0017](../../docs/进展报告/0017-T-26111118-通道状态与运动强对照.md)、[同状态与motion表](report/exploration-r7-aligned/tables.md)、[过程](process/exploration-r7/journal.md)
- 第六轮：[进展报告0016](../../docs/进展报告/0016-T-26111117-局部噪声与RGB通路辨识.md)、[条件矩与RGB表](report/exploration-r6/tables.md)、[机制图](report/exploration-r6/noise_mechanism.png)、[过程](process/exploration-r6/journal.md)
- 第五轮：[进展报告0015](../../docs/进展报告/0015-T-26111116-因果骨干与同容量反证.md)、[完整控制表](report/exploration-r5-combined/tables.md)、[历史替代图](report/exploration-r5-combined/history_mechanism.png)、[过程](process/exploration-r5/journal.md)
- 持续目标：[远期论文目标记录](docs/long-term-goal.md)
- 第四轮：[进展报告0014](../../docs/进展报告/0014-T-26111115-真实数据覆盖与时序强基线.md)、[两域结果](report/exploration-r4/tables.md)、[逐片增量](report/exploration-r4/paired_clip_deltas.json)、[完整过程](process/exploration-r4/journal.md)
- 第三轮：[进展报告0013](../../docs/进展报告/0013-T-26111102-骨干时间交互与真实视频反例.md)、[受控训练表](report/exploration-r3-training/training_tables.md)、[实拍样例](report/exploration-r3-training/real_tables.md)
- 第二轮：[进展报告0012](../../docs/进展报告/0012-T-26111021-训练对照与独立场景探索.md)、[训练/状态表](report/exploration-r2/tables.md)、[训练+融合强基线](report/exploration-r2-fusion/tables.md)
- 首轮结果：[进展报告0010](../../docs/进展报告/0010-T-26111018-首轮调研与预实验.md)、[表格与图](report/exploration-r1/tables.md)
- 上游：共享代码与权重在项目根目录 [`MobileIE/`](../../MobileIE)
- 前置工作：[2 号复现](../2.reproduction/README.md)、[3 号闪烁基线](../3.video-flicker/README.md)、[5 号前期调研](../5.prior-art-survey/README.md)
- 进展报告：[0007-T-261105-4 号工作启动与文献核查](../../docs/进展报告/0007-T-261105-4号工作启动与文献核查.md)

## 1. 本工作的判断

3 号工作保存的历史结果为：合成视频叠加噪声与压缩后，MobileIE 输出残差 **18.07/255**、输入放大 **5.5×**；旧实现的光流 EMA 为 **4.83**。2026-11-10 代码核查发现平移对齐保留回卷边界、光流方向与采样不匹配，以及有损缓存等问题；这些数字保留作历史记录，正式对照需修正后以新实验身份重跑。它们不代表真实视频的正式结论。

本工作必须回答：

1. **后处理是否已经够用？** —— 需要给出后处理在快速运动、遮挡/视差、延迟约束下的失效边界；
2. **原模型哪些结构与训练假设限制视频处理？** —— 用原论文/代码核查、受控实验与等预算消融定位。
3. **训练或结构重设计能否改善画质—稳定性—效率权衡？** —— 与普通去噪、时序损失、后处理和视频增强模型比较，以可重复收益支撑论文贡献。

2026-11-10首轮探索已保存891条方法—序列评估（54种不同输入条件，3个已有场景）。固定光照下“低维门控状态+运动补偿”有局部互补收益，但目标光照变化时固定组合劣于单用运动补偿；输入固定双边滤波也出现明显画质损失。下一轮继续调研和预实验，探索状态适应、噪声/特征变换及可靠时间冗余，不锁定唯一方法或组合。所有结果仍是受控合成预实验。

第二轮增加6张训练/6张验证素材、12次400步受控微调及光照/遮挡/切换条件。更强时序训练可接近当前有界FST收益，后者增加约35%的A100 eager前向时间，因而降低该具体实现优先级。新增普通训练+时间融合强基线，并保留特征噪声相关性/跨帧交互为待验证结构假设；数学微验证不是模型效果。共享GPU启动器会拒绝占用卡，本轮发生过一次真实资源拒绝并换空闲卡执行。

第三轮完成8次同预算微调、FST真实特征统计、门控/输出状态解耦及真实视频样例探索。body交互在原合成集配合相同融合约有0.3dB增量，但真实样例没有迁移；直接低光FB光流也出现高置信错配。当前优先深化真实训练数据、运动可靠性与曝光响应，再扩大骨干候选。增强域光流改善了时间指标但仍有细节/计算代价，不把固定融合当作已解决方案。

第四轮扩大到12训练/6测试片段，完成14次400步训练及数据覆盖、混合梯度、尺度/映射诊断。少量实拍微调的过拟合可由更多覆盖部分缓解，但均值提升不均匀，旧合成微调仍是强参考。训练选alpha0.1及固定半尺度motion将本机串行成本约82.6→34.4ms，必须纳入骨干候选对照；固定photo/census/启发式noise-scale门控未统一解决两域问题。OLiVES为白天ND衰减实拍，不等同自然夜景。

第五轮及两项必要跟进完成20次1000步训练、38条件上的3040模型评价。空间post相对plain实拍+.677dB，历史post相对同位空间post仅+.00023dB；历史替代和合成运动时序监督均未建立该分支的独立历史收益。固定TSM在切换首帧明显退化，时序训练也有跨域质量代价。降低这些具体历史实现优先级，下一步分解FST条件噪声矩、均值偏移与波动，保留强空间/廉价后处理基线。全部仍属探索。

第六轮以24静态条件、独立64校准/64评价draw，验证局部矩与RGB协方差精确分解。body二阶局部能量约68%，经门控/tail后的主要RGB代数分量却是body线性随机响应；保持均值修正减少局部方差仍未统一降低RGB波动。下一步在训练素材核验有效噪声通路及通道选择，避免按大特征能量盲目改平方算子；该诊断尚不能推广到运动或自然夜景。

## 2. 当前产出

| 产出 | 位置 | 说明 |
| --- | --- | --- |
| 第六轮条件矩与RGB通路辨识 | [R6](docs/exploration-r6-protocol.md)、[R6b](docs/exploration-r6b-protocol.md)、[R6c](docs/exploration-r6c-protocol.md)、[结果](report/exploration-r6/)、[过程](process/exploration-r6/) | 独立oracle、局部/输出噪声分离、完整门控/截断协方差及实际资源拒绝 |
| 第五轮因果骨干与同容量反证 | [R5协议](docs/exploration-r5-protocol.md)、[R5b](docs/exploration-r5b-protocol.md)、[R5c](docs/exploration-r5c-protocol.md)、[完整结果](report/exploration-r5-combined/)、[过程](process/exploration-r5/) | 20000更新、历史替代/关闭、同位空间控制、事件响应、源码/GPU审计 |
| 第四轮数据与廉价时序强控制 | [协议](docs/exploration-r4-protocol.md)、[结果/图](report/exploration-r4/)、[过程](process/exploration-r4/) | 5600更新、相同混合输入梯度控制、固定test、传统可靠性与完整处理成本 |
| 第三轮机制与真实视频反例 | [协议](docs/exploration-r3-protocol.md)、[特征/状态图](report/exploration-r3/)、[训练与真实样例](report/exploration-r3-training/)、[过程](process/exploration-r3/) | 相同初始化/信息量控制、history_off、真实低光样例、运动可靠性及状态成本 |
| 第二轮训练与广度验证 | [协议](docs/exploration-r2-protocol.md)、[训练表](report/exploration-r2/)、[融合表](report/exploration-r2-fusion/)、[过程](process/exploration-r2/) | 同初始化、同数据流训练；强正则/后处理控制；成本、反例与GPU准入审计 |
| 首轮调研与预实验 | [协议](docs/exploration-r1-protocol.md)、[图表](report/exploration-r1/)、[过程记录](process/exploration-r1/) | 有效对齐/反向warp校验、单项及组合矩阵、正负结果与特征探针 |
| 架构与视频研究设计 | [`docs/architecture-and-video-research-plan.md`](docs/architecture-and-video-research-plan.md) | FST/HDPA/重参数化、warm-up/IWO、结构候选与强基线矩阵 |
| 文献与新颖性核查 | [`docs/literature-and-novelty.md`](docs/literature-and-novelty.md) | MobileIE 引用追踪 + 相邻工作 + 选题定位 |
| 评价协议（草案） | [`docs/evaluation-protocol.md`](docs/evaluation-protocol.md) | 数据集、指标、成功判据（待导师确认） |
| 模块消融方案 | [`docs/ablation-plan.md`](docs/ablation-plan.md) | 逐模块控制变量实验矩阵 |
| 消融变体工厂 | [`scripts/model_variants.py`](scripts/model_variants.py) | 6 个变体，已通过 slim 等价性冒烟测试 |
| 方法候选与方案 A | [`docs/method-candidates.md`](docs/method-candidates.md) | 三条路线、损失设计、待定问题 |
| 时序损失原型 | [`scripts/temporal_loss.py`](scripts/temporal_loss.py) | flow warp + mask + detach，冒烟测试通过 |
| 闪烁可见性/频谱分析 | [`scripts/flicker_visibility.py`](scripts/flicker_visibility.py)、[`results/visibility/`](results/visibility) | 像素级残差、平坦/纹理区、时间标准差、时间频谱 |
| 复跑的真实退化实验 | [`results/rerun/`](results/rerun) | 用于分析的缓存帧与指标（60 帧，CPU） |

## 3. 可见性分析初步结论（2026-11-05）

针对真实退化条件（泊松噪声 + 读出噪声 + 逐帧 JPEG）：

| 指标 | 输入 | MobileIE 输出 | 放大 |
| --- | --- | --- | --- |
| 像素对齐帧差均值 | 3.23 | **18.24** | 5.6× |
| 像素对齐帧差中位数 | 2.33 | **15.33** | — |
| p95 / p99 | 9.0 / 15.0 | **42.7 / 56.3** | — |
| 超过 8/255 的像素比例 | 6.7% | **77.0%** | — |
| 超过 16/255 的像素比例 | 0.7% | **47.2%** | — |
| 逐像素时间标准差 | 2.69 | **15.14** | 5.6× |
| 平坦区 / 纹理区残差 | 3.63 / 2.90 | **18.14 / 18.56** | — |
| 时间噪声 >3 Hz 功率占比 | 80.5% | **80.4%** | — |

结论：

1. 闪烁**不是少数像素的离群值**，而是空间上普遍存在（77% 像素超过 8/255）；
2. 平坦区与纹理区同样严重——平坦区抖动在视觉上更易被察觉，说明“只处理纹理”的思路不够；
3. 时间噪声呈**宽带**（80% 功率在 3 Hz 以上、主导频率接近 Nyquist），符合“逐帧随机噪声被网络放大”，而不是慢速亮度脉动；
4. 这解释了导师“肉眼看视频不够明显”的观感：残差虽大，但空间上随机、时域上高频，需要帧差放大/局部裁剪/时序曲线才能稳定展示。

图表：`results/visibility/{series,hist,temporal_std_maps,spectrum}.png`。

## 4. 工作计划

以下保留2026-11-05的阶段计划；当前属于前期探索，实际优先级及下一轮假设以最新进展报告0016为准，暂不把方法候选当成已确定的正式实验路线。

### P0（本周，服务器到位前）
- [x] MobileIE 引用追踪与相邻工作检索；
- [x] 评价协议草案、模块消融方案；
- [x] 基于现有数据的可见性/频谱分析；
- [ ] 与导师确认“解决”的量化判据；
- [ ] 真实视频素材收集（自拍 + 公开数据集，待用户提供/确认）。

### P1（11/7 服务器到位后 2–3 周）
- [ ] 迁移到 8×A100 服务器，完成正式训练 baseline；
- [ ] 后处理边界实验：速度扫描、遮挡/视差、延迟；
- [ ] 执行逐模块消融（见 `docs/ablation-plan.md`）。

### P2（12 月，方法原型）
- [x] 方案 A 损失原型（推理零开销，冒烟测试通过）；
- [ ] 方案 A 正式训练与 λ 调参；
- [ ] 方案 B：轻量帧间模块（feature EMA / 1–2 帧输入）；
- [ ] 方案 C：输入多帧拼接（对照）；
- [ ] 统一比较闪烁、画质、参数量、MACs、延迟、快速运动鲁棒性。

### P3（1–2 月，收尾）
- [ ] 手机端 TFLite / Android 模拟器验证；
- [ ] 论文实验与写作（目标 ICCV 2027，3 月 8 日截止）。

## 5. 复现

```bash
PY=F:/dev-tools/python-envs/MobileIE-Neo/Scripts/python.exe

# 重跑真实退化实验（生成缓存帧 + 指标）
"$PY" research/3.video-flicker/scripts/flicker_realistic.py --device cpu \
  --out "F:/资源夹/科研/SQS Team/MobileIE-Neo/research/4.temporal-consistency/results/rerun"

# 可见性 / 频谱分析（依赖上面的缓存帧）
"$PY" research/4.temporal-consistency/scripts/flicker_visibility.py
```

> 缓存帧写在 `F:/pip-tmp/mobileie_frames_real`（ASCII 路径，规避中文路径问题）。

## 6. 修订记录

- 2026-11-05 · claude code (local)：创建 4 号工作，完成文献核查、评价协议、消融方案与可见性分析。
- 2026-11-10 18:03 +08:00 · Codex  (sqs_A100)：以论文目标更新主线入口，标注历史指标待重核，并将结构设计、训练机制和完整基线纳入研究问题。
- 2026-11-10 18:29 +08:00 · Codex  (sqs_A100)：记录首轮自主探索已完成，连接逐序列结果、失败资料和进展报告，保持多路线及组合开放。
- 2026-11-10 21:13 +08:00 · Codex  (sqs_A100)：连接第二轮受控训练、独立素材验证、强基线与资源审计；据成本和对照结果调整候选优先级。

修改：2026-11-11T16:40:43.250064+08:00；作者：Codex  (sqs_A100)；原因：更新0015结果与持续研究入口，历史阶段计划保留供追溯。

修改：2026-11-11T17:09:03.253353+08:00；作者：Codex  (sqs_A100)；原因：索引0016及已完成噪声诊断。

修改：2026-11-11T18:25:49.244614+08:00；作者：Codex  (sqs_A100)；原因：索引0017/0018及优化强对照研究。

修改：2026-11-11T18:37:56.584049+08:00；作者：Codex  (sqs_A100)；原因：索引0019强优化基线，限定低LR结构结论范围。

修改：2026-11-11T18:51:23.382324+08:00；作者：Codex  (sqs_A100)；原因：索引0020原生诊断，记录R11强优化历史评价进行中。

修改：2026-11-11T19:15:58.253077+08:00；作者：Codex  (sqs_A100)；原因：索引0021/0022，记录R13前置历史评价中。

修改：2026-11-11T19:42:53.736037+08:00；作者：Codex  (sqs_A100)；原因：索引0023及R14深度训练启动，保留完整研究目标。

修改：2026-11-11T20:24:45.731053+08:00；作者：Codex  (sqs_A100)；原因：索引R14/R15完整结果及新的结构/数据问题。

修改：2026-11-11T20:48:02.482890+08:00；作者：Codex  (sqs_A100)；原因：索引R16结构训练与R17数据获取限制。

修改：2026-11-11T21:13:08.692013+08:00；作者：Codex  (sqs_A100)；原因：索引0026与固定配对诊断，记录阶段推送已恢复。

修改：2026-11-11T21:21:00.783613+08:00；作者：Codex  (sqs_A100)；原因：索引0027几何负结果与后续时间冗余问题。

修改：2026-11-11T21:48:32.613420+08:00；作者：Codex  (sqs_A100)；原因：索引0028及动态聚合强对照，保持论文目标进行中。

修改：2026-11-11T22:09:32.987886+08:00；作者：Codex  (sqs_A100)；原因：索引0029动态强控制与R21聚合感知训练。

- [0030 聚合训练完成、评价等待](../../docs/进展报告/0030-T-26111122-聚合感知训练完成与共享资源等待.md)：R21完整32000更新/32检查点审计通过；共享GPU被占用，5888行评价待空闲，不作质量声明。

修改：2026-11-11T22:29:57.739588+08:00；作者：Codex  (sqs_A100)；原因：索引R21真实完成状态、资源等待与后续机制诊断。

[R22/报告0031](../../docs/进展报告/0031-T-26111122-FST非线性与插值交换误差诊断.md)：24固定CPU算子诊断通过，非质量/新方法结论；R21全部候选与强参考已切换共同CPU评价，进行中。

修改：2026-11-11T22:35:57.291566+08:00；作者：Codex  (sqs_A100)；原因：索引已完成R22与R21评价后备状态。

[0032 聚合训练后静态强对照](../../docs/进展报告/0032-T-26111122-聚合训练后的静态信息利用强对照.md)：384同CPU统计/80精确重放完成；历史利用与优于经典基线分开，动态R21仍评价中。

修改：2026-11-11T22:50:29.287295+08:00；作者：Codex  (sqs_A100)；原因：索引固定静态终点诊断与负结果。
