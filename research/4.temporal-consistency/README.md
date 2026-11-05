# 4. 时序一致性方法（Temporal Consistency）

目标：在 3 号工作（闪烁现象已复现）的基础上，定义可验收的评价体系，设计并验证**符合手机实时/极轻量约束**的时序一致性方案。

- 状态：**启动**（2026-11-05），P0 推进中
- 上游：共享代码与权重在项目根目录 [`MobileIE/`](../../MobileIE)
- 前置工作：[2 号复现](../2.reproduction/README.md)、[3 号闪烁基线](../3.video-flicker/README.md)、[5 号前期调研](../5.prior-art-survey/README.md)
- 进展报告：[0007-T-261105-4 号工作启动与文献核查](../../docs/进展报告/0007-T-261105-4号工作启动与文献核查.md)

## 1. 本工作的判断

3 号工作已量化：真实退化下 MobileIE 输出的对齐帧间残差 **18.07/255**、输入放大 **5.5×**；光流 EMA 后处理可降到 **4.83**。因此本工作必须回答两个问题：

1. **后处理是否已经够用？** —— 需要给出后处理在快速运动、遮挡/视差、延迟约束下的失效边界；
2. **训练侧方案能否在零/极低推理开销下做得更好？** —— 这是论文的立身之本。

## 2. 当前产出

| 产出 | 位置 | 说明 |
| --- | --- | --- |
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
