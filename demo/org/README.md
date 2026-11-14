# 6. 手机端部署与现场演示（Mobile Deployment & Demo）

目标：为项目申报的现场考察，按 MobileIE 官方流程把模型跑在安卓手机上，并提供“手机实测速度 + 处理前后效果对比”的演示材料。**这是演示任务，不是研究任务**，不改变 3/4/5 号工作的技术路线。

- 分支：**`demo.org`**；本目录 `demo/org/` 是这项演示工作的主文件夹。`main` 单向同步到本分支，演示改动不合并回 `main`
- 状态：**P1 完成**（2026-11-12，备用机一加 9R 跑通自定义模型：GPU FP16 **5.8 ms**、CPU **86–91 ms**）；**微光实验室 0.1.1 已于 2026-11-13 在一加 9R 真机跑通**（拍照推理 106.7 ms，见 [10 真机检查](docs/10-real-device-0.1.1.md)），现场演示准备待做
- 验收安排（2026-11-13 用户确认）：**2026-11-16（下周一）上午最终验收**；导师提前检查演示，用户主要协助演示。预检查时间、验收具体时刻与地点待确认，见[预检查与验收清单](docs/04-acceptance-checklist.md)。
- 当前交付（2026-11-13）：**Android 0.1.2 状态刷新修复版 / Python 0.1.1**，支持现场拍照（Android）、选图、MobileIE/Gamma 对照、滑动/总览和保存。[个人安卓环境报告](docs/06-android-environment.md)已先行提交。0.1.1 的一加 9R 检查见[10 真机记录](docs/10-real-device-0.1.1.md)；其中样例失效的同症状回调缺陷已受控复现并修正，17 项模拟回归通过，安装包与说明见[11 修复报告](docs/11-android-0.1.2-state-fix.md)。**0.1.2 已于一加 9R 完成同机复验**（见[12 真机复验](docs/12-android-0.1.2-reverify.md)）；导师预检查尚未完成（Windows 桌面版于 2026-11-13 按用户决定不再作为验收项，仅归档备查）。
- 上游代码与权重：项目根目录 [`MobileIE/`](../../MobileIE)
- 关联：[2 号复现](../../research/2.reproduction/README.md)、[5 号前期调研](../../research/5.prior-art-survey/README.md)、[6B06-01 会议记录](../../docs/会议记录/6B06-01/audio%20-%20Transcription.txt)
- 进展报告：[0009-T-261108-6 号工作启动与手机演示方案](../../docs/进展报告/0009-T-261108-6号工作启动与手机演示方案.md)

**协作约定**

- 本仓库是唯一数据站：日志、结论、截图、脚本等全部入库，不依赖本地或服务器临时文件；
- 小步、密集提交，阶段性 `git push origin demo.org`；
- 同步 `main` 只用 `git fetch origin` + `git merge --no-ff origin/main`，**禁止 `git pull`**，以保留完整合并历史；`main` 由 Codex 在服务器上快速迭代，合并前先看 `git log --oneline HEAD..origin/main`。
- 用户于 2026-11-13 重申：**严禁 fast-forward，所有合并必须 `--no-ff`**。推送因远端分歧失败时，先 fetch，以 `--no-ff` 合并并解决冲突后重试；不 rebase、squash 或强推。

## 1. 任务边界（先看这里）

导师在 2026-11-08 的会议中要求：尽快在安卓手机上把论文里的 **AI Benchmark** 跑起来并演示，用于项目申报的现场考察；不用做研究，照原文流程即可。

核对论文、官方 README 与代码后确认：

1. 论文的手机部分**只用于测推理延迟**（Snapdragon 8 Gen 3，0.895 ms / 1100 FPS），没有手机 App，也没有可视化 demo 代码；
2. 官方 README 给出的手机流程是：`lolv1.tflite` → 导入 **AI Benchmark** → 得到推理速度；
3. 因此“论文原做法”只能给速度，给不了画面。

**历史决策（2026-11-08）**：当时采用 **B 方案**：

- **手机端（严格照论文）**：AI Benchmark v7 PRO Mode 加载 `lolv1.tflite`，现场展示实测延迟；
- **效果端（复用官方脚本）**：在电脑上运行仓库自带的 `MobileIE/test_TFLite_RGB.py`，用同一份 TFLite 模型生成低光增强前后对比图，放到手机/投影上展示。

现场表述：同一份 TFLite 模型，手机实测 X ms；效果对比由官方推理脚本输出。

**2026-11-13 需求更新**：用户先要求设计，随后授权环境、报告、App、模拟测试与自主迭代。双端 0.1.1 交付见[09 报告](docs/09-app-v011-final.md)；当前 Android 修复版为[0.1.2](docs/11-android-0.1.2-state-fix.md)。真机彩排通过后采用微 App 图片效果 + AI Benchmark 独立测速；上述 B 方案保留备用。

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

## 3. P0 跑通结果（2026-11-08）

- 官方 `test_TFLite_RGB.py`（经 [`scripts/run_official_tflite.py`](scripts/run_official_tflite.py) 包装）跑完 LOLv1 eval15 全部 15 张；
- **PSNR 均值 23.619 dB**，与论文报告的 23.62 dB 一致，说明模型与预处理链路正确；
- 对比图：[`results/compare_lol_eval15/`](results/compare_lol_eval15)（15 张）、[`results/compare_sample/`](results/compare_sample)（仓库样例）；
- 样例与结论见 [HTML 报告](report/mobile-deployment.html)。

**环境说明**：本机 `tensorflow-cpu` 2.20/2.21 原生库初始化失败（DLL 初始化错误），电脑侧改用官方 LiteRT 运行时 `ai-edge-litert` 2.3.0，由包装脚本注入 `tf.lite.Interpreter` 兼容层；该 runtime 的默认 XNNPACK delegate 在本机 `allocate_tensors` 时段错误，故关闭默认 delegate、使用纯内置算子。官方脚本的预处理/推理/保存逻辑未改动；手机端仍使用原始 `.tflite` 模型。

## 4. 产出

| 产出 | 位置 | 状态 |
| --- | --- | --- |
| 演示方案与任务拆解 | [`docs/01-demo-plan.md`](docs/01-demo-plan.md) | 完成 |
| AI Benchmark 手机操作手册 | [`docs/02-ai-benchmark-guide.md`](docs/02-ai-benchmark-guide.md) | 完成 |
| 官方 TFLite 脚本使用说明 | [`docs/03-official-tflite-inference.md`](docs/03-official-tflite-inference.md) | 完成 |
| 导师预检查与验收清单 | [`docs/04-acceptance-checklist.md`](docs/04-acceptance-checklist.md) | 已整理，真机彩排与导师反馈待完成 |
| 拍照增强与传统对照微 App 设计 | [`docs/05-micro-app-design.md`](docs/05-micro-app-design.md) | Android 0.1.2 已实现，模拟验收通过 |
| 个人安卓开发环境报告 | [`docs/06-android-environment.md`](docs/06-android-environment.md) | 工具、编译签名、AVD 启动及安装门控通过 |
| 自主交付门控 | [`docs/07-delivery-gates.md`](docs/07-delivery-gates.md) | 实施与可靠验收的结束条件 |
| 双端首轮交付及验收报告 | [`docs/09-app-v011-final.md`](docs/09-app-v011-final.md) | 0.1.1 安装包、Python ZIP、录像与证据已交付 |
| Android 状态刷新修复报告 | [`docs/11-android-0.1.2-state-fix.md`](docs/11-android-0.1.2-state-fix.md) | 0.1.2 APK、受控复现及 17 项通过记录 |
| 0.1.2 一加 9R 复验 | [`docs/12-android-0.1.2-reverify.md`](docs/12-android-0.1.2-reverify.md)、[`results/phone-0.1.2/`](results/phone-0.1.2) | 完成（2026-11-13，覆盖升级与多轮往返选样例通过） |
| 模型结构解析脚本 | [`scripts/inspect_tflite.py`](scripts/inspect_tflite.py) | 完成 |
| 演示图片预处理脚本 | [`scripts/prepare_demo_images.py`](scripts/prepare_demo_images.py) | 完成 |
| 官方推理脚本包装 | [`scripts/run_official_tflite.py`](scripts/run_official_tflite.py) | 完成 |
| 模型 I/O 解析结果 | [`data/model_io.txt`](data/model_io.txt) | 完成 |
| 测试设备参数 | [`data/test-devices.md`](data/test-devices.md) | 完成（一加 13 / 一加 9R） |
| 前后对比图 | [`results/compare_lol_eval15/`](results/compare_lol_eval15)、[`results/compare_sample/`](results/compare_sample) | 完成 |
| 增强结果 | [`results/lol_eval15/`](results/lol_eval15)、[`results/demo_output_sample/`](results/demo_output_sample) | 完成 |
| 手机实测截图/录屏 | [`results/phone/`](results/phone) | 完成（启动、标准跑分、自定义模型 8 张后端截图） |
| 微光实验室真机检查 | [`results/phone-0.1.1/`](results/phone-0.1.1)、[10 真机检查](docs/10-real-device-0.1.1.md) | 完成（2026-11-13，一加 9R；含 1 个待修复问题） |
| 手机演示包 | [`demo-package/MobileIE-Demo/`](demo-package/MobileIE-Demo/)（本地压缩件 `demo-package/MobileIE-手机演示包.zip`） | 完成（模型 + 3 张对比图 + 中文操作说明） |
| 汇报 HTML | [`report/mobile-deployment.html`](report/mobile-deployment.html) | 完成（含 P0 结果，单文件内嵌图片） |

## 5. 执行清单

### P0 —— 电脑侧效果（已完成）
- [x] 建立工作目录、方案与文档；
- [x] 解析 TFLite 模型输入输出与算子；
- [x] 新建独立演示环境 `F:\dev-tools\python-envs\MobileIE-Demo`（TensorFlow 原生库本机不可用，改用 `ai-edge-litert`）；
- [x] 用官方 `test_TFLite_RGB.py` 跑通 `lolv1.tflite`，LOL eval15 PSNR 23.619 dB；
- [x] 用 `prepare_demo_images.py` 处理仓库暗光样例，生成 400×600 输入与增强结果；
- [ ] 用手机实拍暗光照片替换样例，生成最终演示素材（待用户提供照片）。

### P1 —— 手机侧速度（已完成，一加 9R）
- [x] 手机安装 AI Benchmark（官网直连 APK，1.98 GB）；
- [x] `model.tflite` 放入 Download，PRO Mode → Custom Model 加载；
- [x] 记录 CPU / NNAPI / GPU / QNN 等后端的延迟，截屏（`results/phone/`）。

**实测结论**（详见 [`data/test-devices.md`](data/test-devices.md) §4 与 [今日日志](journal/2026-11-12-phone-setup.md)）：

| 后端 | 精度 | 单帧延迟 | 说明 |
| --- | --- | --- | --- |
| TFLite GPU Delegate | FP16 | **5.76 / 5.95 ms**（≈170 FPS） | 演示主推；初始化约 450 ms |
| TFLite GPU Delegate | FP32 | 23.6 ms | |
| Qualcomm QNN GPU | FP16 | 15.8 ms | |
| CPU（4 线程） | FP16/FP32 | 86–91 ms | 表头在 CPU 下不区分精度 |
| Android NNAPI | FP16 | 154 ms | 未真正下沉，仅记录 |
| Arm NN | FP16 | 崩溃 | OpenCL 导入输入张量异常，见 `data/armnn-crash.txt` |

### P2 —— 现场演示
- 最终验收定于 **2026-11-16 上午**；导师将在此之前检查，用户主要协助演示。
- [ ] 完成[预检查清单](docs/04-acceptance-checklist.md)，复跑 GPU FP16 并记录本次结果与导师反馈；
- [ ] 按 [`docs/01-demo-plan.md`](docs/01-demo-plan.md) 的演示脚本串场；
- [ ] 准备离线素材：手机本地模型、前后对比图、录屏，避免现场依赖网络。

### P3 —— 微 App（2026-11-13，Android 0.1.2 / Python 0.1.1）
- 需求为系统拍照后单张处理、相册选图、MobileIE / 传统方法 / 原图对比，详见[设计草案](docs/05-micro-app-design.md)。
- [x] 用户授权自主实现，环境报告已提交，门控已登记。
- [x] Android 功能、6 组数值、17 项仪器回归、异常/生命周期、视觉与交付门控通过。
- [x] Python 伴随版及 Linux/Xvfb 完整 5 项测试、三样例精确一致与便携 ZIP 核验通过。
- [x] 安装包、28.837 秒模拟录像、摘要、配置/最终报告和操作说明交付。
- [x] 0.1.1 一加 9R 主要功能实测，保留样例失效问题记录及 0.1.2 受控修复。
- [x] 0.1.2 一加 9R 复验（2026-11-13，见[12](docs/12-android-0.1.2-reverify.md)）；真实 Windows 和导师预检查待接续，不以模拟器结果替代。

## 6. 复现

```bash
PY=F:/dev-tools/python-envs/MobileIE-Neo/Scripts/python.exe
PYD=F:/dev-tools/python-envs/MobileIE-Demo/Scripts/python.exe

# 解析 TFLite 模型（仅标准库）
"$PY" demo/org/scripts/inspect_tflite.py

# 缩放手机照片到 600×400
"$PY" demo/org/scripts/prepare_demo_images.py resize \
  --input photos/ --output demo/org/data/demo_input

# 官方 TFLite 推理（包装脚本自动选择 TensorFlow 或 ai-edge-litert）
"$PYD" demo/org/scripts/run_official_tflite.py \
  --model MobileIE/pretrain/lolv1.tflite \
  --input demo/org/data/demo_input \
  --output demo/org/results/demo_output

# 生成左右对比图
"$PY" demo/org/scripts/prepare_demo_images.py compare \
  --input demo/org/data/demo_input \
  --enhanced demo/org/results/demo_output \
  --output demo/org/results/compare_demo \
  --format jpeg
```

## 7. 风险与待确认

1. **现场预期**：AI Benchmark 只给延迟；微 App 提供拍照后单张处理，未实现摄像头实时增强取景；
2. **模型输入固定 400×600**：手机原图必须预处理，否则无法推理；
3. **依赖环境**：本机 `tensorflow-cpu` 2.20/2.21 原生库初始化失败，已改用 `ai-edge-litert` + 关闭默认 XNNPACK delegate；换机器时需重新验证；
4. **AI Benchmark**：v4 起支持自定义模型，v7.0.0（2026-10）仍保留；需跑完标准测试后进入 PRO Mode，APK 较大；国内网络下优先用官网直连 APK；
5. **模型任务**：`lolv1.tflite` 是低光增强模型，现场素材应选暗光照片；水下增强只有 `.pkl`，需另行转换 TFLite。
6. **AI Benchmark 操作坑**：CPU 下选 FP16/FP32 运行页表头都写 `FP16`（不区分）；INT8 档位要求量化模型，FP32 模型直接报 `IllegalArgumentException`；Arm NN 后端会崩溃并清空全部设置；切换精度会把后端重置回 CPU。现场固定“**FP16 + TFLite GPU Delegate**”，选完先看一眼表头再点 Run。

## 8. 修订记录

- 2026-11-13 · claude code (local)：一加 9R 安装微光实验室 0.1.1，完成样例/拍照/相册/对比/保存/AI Benchmark 跳转真机检查（拍照推理 106.7 ms），记录 1 个待修复的样例选择失效问题。
- 2026-11-13 · claude code (local)：`--no-ff` 合入 0.1.2 修复；一加 9R 覆盖升级保留数据，根因链与相机/选择器/AI Benchmark 多轮往返后连续选样例均通过，见 [12 复验](docs/12-android-0.1.2-reverify.md)。
- 2026-11-13 · Codex  (sqs_A100)：补充拍照、相册选图和 Gamma 传统对照的微 App 设计，记录只写文档、暂不制作的用户要求；保留旧演示方案为备份。
- 2026-11-13 · Codex  (sqs_A100)：接手 `demo.org`；记录周一上午最终验收、导师提前检查及用户协助演示分工，新增预检查清单，修订演示包操作顺序，重申禁止 fast-forward 与阶段推送约定。此次仅完成文档准备，未进行新一轮真机测试。
- 2026-11-08 · claude code (local)：创建 6 号工作，确定 B 方案，完成模型解析与文档首版。
- 2026-11-08 · claude code (local)：P0 跑通，LOL eval15 PSNR 23.619 dB；补充 ai-edge-litert 兼容层与结果对比图。
- 2026-11-10 · claude code (local)：建立 `demo.org` 演示分支；打包手机演示包（模型 + 对比图 + 中文操作说明）。
- 2026-11-12 · claude code (local)：工作目录迁入 `demo/org/`；与 `main` 做首次 `--no-ff` 同步（2382 个提交）。
- 2026-11-12 · claude code (local)：备用机完成 P1 多后端测速（GPU FP16 5.8 ms / CPU 90 ms），记录 INT8 不兼容与 Arm NN 崩溃；与 `main` 二次同步（281 个提交）。
