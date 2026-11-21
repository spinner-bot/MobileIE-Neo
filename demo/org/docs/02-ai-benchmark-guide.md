# 02 · AI Benchmark 手机操作手册

- 创建：2026-11-08 · claude code (local)
- 触发：导师要求按论文流程在安卓手机上演示 MobileIE 的端侧推理速度。
- 关联：[演示方案](01-demo-plan.md)、[官方 TFLite 脚本说明](03-official-tflite-inference.md)
- 依据：AI Benchmark 官方发布说明（v4 起支持自定义 TFLite 模型）、v7.0.0 更新记录、`MobileIE/README.md`

## 1. 工具概况

| 项 | 内容 |
| --- | --- |
| 名称 | AI Benchmark（包名 `org.benchmark.demo`） |
| 开发者 | Andrey Ignatov / ETH Zurich |
| 官网 | <https://ai-benchmark.com/download> |
| 当前版本 | v7.0.0（2026-10 更新；LiteRT 1.4.2） |
| 平台 | Android 5.0+；**不支持 iOS** |
| 体积 | 约 1.98 GB（安装包 + 解压后更大，需预留 5 GB） |
| 关键功能 | PRO Mode → Custom Model，可加载任意 TFLite 模型并选择后端/精度 |

> 注意：它是**性能跑分 App**，不是“模拟器”；自定义模型模式只输出延迟/吞吐和错误日志，**不显示模型输出图像**。

## 2. 安装

### 2.1 首选：官网直连 APK

1. 手机浏览器打开 <https://ai-benchmark.com/download>；
2. 找到 “AI Benchmark” 最新版（v7.0.0）的 “Direct APK Download”；
3. 允许浏览器“安装未知应用”，下载并安装；
4. 首次打开授予存储权限（读取 Download 中的模型需要）。

### 2.2 备选：Google Play

- 应用 ID：`org.benchmark.demo`；
- 国内设备若无 GMS，优先用官网 APK。

### 2.3 Nightly（仅当正式版不兼容模型时）

- 官网提供 Nightly 版，使用最新 TFLite runtime，支持更多算子；
- 官方说明：仅供研发，不保证设备兼容性，不作为正式跑分。

## 3. 导入 MobileIE 模型

### 3.1 准备模型

- 使用仓库现成文件：[`MobileIE/pretrain/lolv1.tflite`](../../MobileIE/pretrain/lolv1.tflite)（23 KB，FP32，输入 1×400×600×3）；
- 复制到手机 **Download** 目录，并命名为 **`model.tflite`**（App 默认查找该文件名）；
- 传输方式：USB 文件传输、微信文件传输助手、adb push 均可。

```bash
# 可选：adb 传输（需开启 USB 调试）
adb push MobileIE/pretrain/lolv1.tflite /sdcard/Download/model.tflite
```

### 3.2 App 内操作顺序

1. 打开 AI Benchmark，先**完整跑一次标准测试**（部分版本需完成后才显示 PRO Mode 入口）；
2. 进入 **PRO Mode**；
3. 找到 **Custom Model**（自定义模型）页；
4. 模型来源二选一：
   - 自动：确认 `Download/model.tflite` 存在，直接选择；
   - 手动：点击选择文件，在文件管理器中开启 **“显示内部存储”**，选中完整路径下的 `model.tflite`；
5. 选择精度：本模型为 FP32，优先选 **FP32**（若界面要求 INT8/FP16，先跳过，见 §6）；
6. 选择后端并逐个运行：
   - TFLite CPU / XNNPACK（最稳，先跑这个）；
   - Android NNAPI；
   - TFLite GPU delegate；
   - Qualcomm QNN / Hexagon、MediaTek Neuron、Samsung ENN（视机型）；
7. 记录每个后端的延迟（ms）与 FPS，截屏或录屏。

## 4. 模型兼容性核对（已完成）

| 要求 | `lolv1.tflite` | 结论 |
| --- | --- | --- |
| 单输入 / 单输出 | 是 | ✅ |
| 固定 4D 输入 | 1×400×600×3 | ✅ |
| 标准算子 | 23 个，无自定义 op | ✅ |
| 数据类型 | FP32 | 选 FP32 后端 |
| 文件大小 | 23 KB | ✅ |

模型解析命令与结果见 [`../scripts/inspect_tflite.py`](../scripts/inspect_tflite.py)、[`../data/model_io.txt`](../data/model_io.txt)。

## 5. 记录表（现场/汇报用）

| 日期 | 手机型号 | Android | App 版本 | 后端 | 精度 | 延迟 (ms) | FPS | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 待填 | 待填 | 待填 | 待填 | CPU/XNNPACK | FP32 | | | |
| 待填 | 待填 | 待填 | 待填 | NNAPI | FP32 | | | |
| 待填 | 待填 | 待填 | 待填 | GPU | FP32 | | | |

> 论文参考值：Snapdragon 8 Gen 3，0.895 ms（约 1117 FPS）。实际机型/后端/温度不同，数值会明显差异，汇报时注明机型。

## 6. 常见问题

| 现象 | 处理 |
| --- | --- |
| 找不到 PRO Mode / Custom Model | 先完整跑一次标准测试；确认 App 为 v4 以上（建议 v7） |
| 看不到 Download 里的模型 | 授予存储权限；文件管理器开启“显示内部存储”；确认文件名是 `model.tflite` |
| 加载报算子不支持 | 用 Nightly 版；或确认模型未被量化/改过 |
| 选 INT8/FP16 报错 | 本模型是 FP32，选 FP32；需要 INT8 须重新转换并做量化校准 |
| 运行卡死/闪退 | 换 CPU 后端；关闭其他后台；重启 App；检查剩余存储 |
| 数值与论文差距大 | 机型、SoC、后端、温度不同，属正常；以同机型同后端对比为准 |

## 7. 现场注意事项

- 提前在**有网环境**完成 App 安装与首次标准测试，现场只做模型加载与运行；
- 准备**离线备份**：模型文件、测速录屏、前后对比图；
- 手机电量充足、关闭省电模式，避免降频影响数值；
- 若现场网络差，不要依赖临时下载 APK。

## 8. 修订记录

- 2026-11-08 · claude code (local)：创建，整理安装、导入、兼容性与排错步骤。
