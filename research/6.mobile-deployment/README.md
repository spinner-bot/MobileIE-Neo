# 6. 手机端部署与现场演示（Mobile Deployment & Demo）

目标：为项目申报的现场考察，按 MobileIE 官方流程把模型跑在安卓手机上，并提供“手机实测速度 + 处理前后效果对比”的演示材料。**这是演示任务，不是研究任务**，不改变 3/4/5 号工作的技术路线。

- 状态：**启动**（2026-11-08）
- 上游代码与权重：项目根目录 [`MobileIE/`](../../MobileIE)
- 关联：[2 号复现](../2.reproduction/README.md)、[5 号前期调研](../5.prior-art-survey/README.md)、[6B06-01 会议记录](../../docs/会议记录/6B06-01/audio%20-%20Transcription.txt)
- 进展报告：[0009-T-261108-6 号工作启动与手机演示方案](../../docs/进展报告/0009-T-261108-6号工作启动与手机演示方案.md)

## 1. 任务边界（先看这里）

导师在 2026-11-08 的会议中要求：尽快在安卓手机上把论文里的 **AI Benchmark** 跑起来并演示，用于项目申报的现场考察；不用做研究，照原文流程即可。

核对论文、官方 README 与代码后确认：

1. 论文的手机部分**只用于测推理延迟**（Snapdragon 8 Gen 3，0.895 ms / 1100 FPS），没有手机 App，也没有可视化 demo 代码；
2. 官方 README 给出的手机流程是：`lolv1.tflite` → 导入 **AI Benchmark** → 得到推理速度；
3. 因此“论文原做法”只能给速度，给不了画面。

**决策（2026-11-08）**：采用 **B 方案**，不新写安卓 App、不造轮子：

- **手机端（严格照论文）**：AI Benchmark v7 PRO Mode 加载 `lolv1.tflite`，现场展示实测延迟；
- **效果端（复用官方脚本）**：在电脑上运行仓库自带的 `MobileIE/test_TFLite_RGB.py`，用同一份 TFLite 模型生成低光增强前后对比图，放到手机/投影上展示。

现场表述：同一份 TFLite 模型，手机实测 X ms；效果对比由官方推理脚本输出。

## 2. 模型事实（已解析 `lolv1.tflite`）

| 项 | 值 |
| --- | --- |
| 文件 | [`MobileIE/pretrain/lolv1.tflite`](../../MobileIE/pretrain/lolv1.tflite) |
| 大小 | 23 KB |
| 输入 | `serving_default_input_1:0`，固定 **1×400×600×3**（NHWC），float32 |
| 输出 | `StatefulPartitionedCall:0`，1×400×600×3，float32 |
| 算子 | 23 个，全部标准算子：CONV_2D ×6、MUL ×9、ADD ×2、LOGISTIC ×2、PAD / MEAN / PRELU / REDUCE_MAX 各 1 |
| 自定义算子 | 无 |
| 量化 | FP32 |

结论：单输入、单输出、固定形状、无自定义 op，AI Benchmark 的 Custom Model 入口应当可以直接加载。输入固定为 400×600，手机照片必须先缩放/裁剪到这个尺寸。

解析脚本：[`scripts/inspect_tflite.py`](scripts/inspect_tflite.py)；原始输出：[`data/model_io.txt`](data/model_io.txt)。

## 3. 产出

| 产出 | 位置 | 状态 |
| --- | --- | --- |
| 演示方案与任务拆解 | [`docs/01-demo-plan.md`](docs/01-demo-plan.md) | 完成 |
| AI Benchmark 手机操作手册 | [`docs/02-ai-benchmark-guide.md`](docs/02-ai-benchmark-guide.md) | 完成 |
| 官方 TFLite 脚本使用说明 | [`docs/03-official-tflite-inference.md`](docs/03-official-tflite-inference.md) | 完成 |
| 模型结构解析脚本 | [`scripts/inspect_tflite.py`](scripts/inspect_tflite.py) | 完成 |
| 演示图片预处理脚本 | [`scripts/prepare_demo_images.py`](scripts/prepare_demo_images.py) | 完成 |
| 官方推理脚本包装 | [`scripts/run_official_tflite.py`](scripts/run_official_tflite.py) | 完成 |
| 模型 I/O 解析结果 | [`data/model_io.txt`](data/model_io.txt) | 完成 |
| 前后对比图 / 演示视频 | `results/` | 待跑 |
| 手机实测截图/录屏 | `results/phone/` | 待采集 |
| 汇报 HTML | [`report/mobile-deployment.html`](report/mobile-deployment.html) | 完成（首版） |

## 4. 执行清单

### P0 —— 电脑侧效果（当前）
- [x] 建立工作目录、方案与文档；
- [x] 解析 TFLite 模型输入输出与算子；
- [ ] 在虚拟环境安装 TFLite 推理依赖（TensorFlow 或 tflite-runtime）；
- [ ] 用官方 `test_TFLite_RGB.py` 跑通 `lolv1.tflite`，生成 LOL 样例前后对比图；
- [ ] 用 `prepare_demo_images.py` 处理手机实拍暗光照片，生成 400×600 输入与增强结果。

### P1 —— 手机侧速度
- [ ] 手机安装 AI Benchmark（官网直连 APK，约 1.4 GB）；
- [ ] `model.tflite` 放入 Download，PRO Mode → Custom Model 加载；
- [ ] 记录 CPU / NNAPI / GPU 等后端的延迟，截屏或录屏。

### P2 —— 现场演示
- [ ] 按 [`docs/01-demo-plan.md`](docs/01-demo-plan.md) 的演示脚本串场；
- [ ] 准备离线素材：手机本地模型、前后对比图、录屏，避免现场依赖网络。

### P3 —— 备选（仅在要求“手机实时处理画面”时启用）
- Termux 跑官方脚本（不写新 App）或最小安卓 App；需重新评估工作量。

## 5. 复现

```bash
PY=F:/dev-tools/python-envs/MobileIE-Neo/Scripts/python.exe

# 解析 TFLite 模型（仅标准库）
"$PY" research/6.mobile-deployment/scripts/inspect_tflite.py

# 手机照片预处理为 400×600（放入 MobileIE/data/samples/lowlight 或自定目录）
"$PY" research/6.mobile-deployment/scripts/prepare_demo_images.py --help

# 官方 TFLite 推理（需先安装 tensorflow；详见 docs/03）
cd MobileIE && "$PY" test_TFLite_RGB.py
```

## 6. 风险与待确认

1. **现场预期**：AI Benchmark 只给延迟。若评审要求“手机摄像头实时增强”，B 方案不满足，需要 P3；
2. **模型输入固定 400×600**：手机原图必须预处理，否则无法推理；
3. **依赖环境**：当前虚拟环境没有 TensorFlow / tflite_runtime，需在 P0 安装；
4. **AI Benchmark**：v4 起支持自定义模型，v7.0.0（2026-10）仍保留；需跑完标准测试后进入 PRO Mode，APK 较大；国内网络下优先用官网直连 APK；
5. **模型任务**：`lolv1.tflite` 是低光增强模型，现场素材应选暗光照片；水下增强只有 `.pkl`，需另行转换 TFLite。

## 7. 修订记录

- 2026-11-08 · claude code (local)：创建 6 号工作，确定 B 方案，完成模型解析与文档首版。
