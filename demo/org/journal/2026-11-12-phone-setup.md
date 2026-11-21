# 2026-11-12 备用机连接与安装日志

- 设备：一加 9R（LE2100），Android 14 / ColorOS 14.0
- 连接方式：USB 连接 Windows，目标用 adb 完成安装、推模型、截图

## 11-12 连接检查

| 检查 | 结果 |
| --- | --- |
| Windows 设备枚举 | 可见 `WPD / OnePlus 9R`（MTP 模式正常） |
| `adb devices` | 空 |
| 判断 | 数据线与 USB 口正常；手机未开启 USB 调试，或调试授权未通过 |

### 待用户操作

1. 设置 → 关于本机 → 版本信息 → 连续点击“版本号”7 次，开启开发者选项；
2. 设置 → 系统设置 → 开发者选项 → 打开 **USB 调试**；
3. 重新插拔数据线，手机弹“允许 USB 调试吗” → 勾选“始终允许” → 允许；
4. USB 用途保持“传输文件”。

## APK 准备

- 官方直链：<https://download.ai-benchmark.com/s/4zWyEM6KD9dgbRk/download/AI_Benchmark_V7.0.0.apk>
- 实际大小：**1,978,394,424 字节（约 1.98 GB）**，此前记录的“约 1.4 GB”偏小，已修正；
- 下载位置：`F:/pip-tmp/aibench/AI_Benchmark_V7.0.0.apk`（仓库外，避免大文件入库）；
- 备用机剩余 71.9 GB，安装空间充足。
- 11-12 01:45 下载完成：字节数与 Content-Length 完全一致，`unzip -l` 可正常列出 2291 个条目，APK 完整。

## 下一步

1. adb 授权通过后确认型号、系统、剩余空间；
2. `adb install` 安装 APK；
3. `adb push demo/org/demo-package/MobileIE-Demo/model.tflite /sdcard/Download/`；
4. 启动 App、截图，检查 Custom Model 入口。

## 11-12 02:20–02:30 安装与启动

| 步骤 | 结果 |
| --- | --- |
| `adb devices` | `b62e7c1 device`，LE2100 / OnePlus / Android 14 / SDK 34 / arm64-v8a |
| 剩余空间 | 66 GB（`/storage/emulated`） |
| `adb install -r` | **Success**，版本 `7.0.0`（versionCode 58，targetSdk 36） |
| 推模型 | `/sdcard/Download/model.tflite`，23,228 字节 |
| 启动 | `org.benchmark.demo` 正常启动，约 8 秒后进入 `IntermediateActivity` |

启动后 App **自动开始完整标准跑分**，首屏为 `Section: 1/30. Recognition`、`MobileNet-V4 Large`、`448×448`。截屏见 `../results/phone/01_launch.jpg`（2 MB PNG 原件留在 `F:/pip-tmp/aibench/`，仓库只存 JPEG）。

按手册，需要等这套 30 段标准测试跑完，PRO Mode / Custom Model 入口才会出现。测试期间保持前台、不锁屏。

## 11-12 02:56 标准测试完成

- SoC AI Score **460.5**，Device AI Score **651.3**（Medium Performance）；
- INT8：AI Speed 194、Accuracy 79.0%；FP16：AI Speed 290、Accuracy 88.2%；
- 截图：`../results/phone/02_standard_result.jpg`。

## 11-12 03:02 自定义模型首次运行（CPU）

- 入口：PRO MODE → Custom Model → 选择 `Download/model.tflite`；
- 模型被正确识别：Input `[1, 400, 600, 3]`，Output `[1, 400, 600, 3]`；
- 加速器：CPU（4 线程）；迭代 10 次；
- **Avg Latency 90.7 ms**，STD 3.17 ms，Initialization 42.3 ms；
- 逐次延迟：129.0 / 101.0 / 105.0 / 88.4 / 98.5 / 90.4 / 91.4 / 88.7 / 96.8 / 88.4 ms；
- 截图：`../results/phone/03_custom_cpu_fp16.jpg`。

**待核对**：运行页表头显示 `FP16`，但界面上已选中 `FP32`；需要用 FP32 重跑并确认表头，避免把精度档位记错。另外输入范围仍是默认 `0,255`，对延迟测量无影响，但后续可改成 `0,1` 与训练预处理一致。

## 11-12 03:04–03:11 多后端实测（一加 9R）

统一条件：`model.tflite`（FP32，输入 `[1,400,600,3]`），10 次迭代，迭代间隔 0 ms，输入范围 `0,255`，室温未控温。

| 后端 | 精度档 | 运行页表头 | Avg（ms） | STD（ms） | Init（ms） | 截图 |
| --- | --- | --- | --- | --- | --- | --- |
| CPU（4 线程） | FP16 | FP16 | 90.7 | 3.17 | 42.3 | `03_custom_cpu_fp16.jpg` |
| CPU（4 线程） | FP32 | FP16 | 89.9 | 5.20 | 46.1 | — |
| CPU（4 线程） | FP32 | FP16 | 86.4 | 1.94 | 47.6 | `04_custom_cpu_fp32.jpg` |
| TFLite GPU Delegate | FP32 | GPU / FP32 | 23.6 | 1.08 | 337.0 | `05_custom_gpu_fp32.jpg` |
| TFLite GPU Delegate | FP16 | GPU / FP16 | **5.76** | 0.89 | 454.0 | `06_custom_gpu_fp16.jpg` |
| TFLite GPU Delegate | FP16（复跑） | GPU / FP16 | **5.95** | 0.87 | 447.0 | `07_custom_gpu_fp16_repeat.jpg` |
| Android NNAPI | FP16 | NNAPI / FP16 | 154.0 | 4.78 | 881.0 | `08_custom_nnapi_fp16.jpg` |
| Qualcomm QNN GPU | FP16 | QNN GPU / FP16 | 15.8 | 0.75 | 863.0 | `09_custom_qnn_gpu_fp16.jpg` |
| Arm NN | FP16 | — | 崩溃 | — | — | `11_armnn_crash.jpg` |
| （INT8 档位） | INT8 | INT8 | 无法运行 | — | — | `10_custom_int8_incompatible.jpg` |

**结论（演示建议）**：TFLite GPU Delegate + FP16 是这台机器上最快且稳定的组合，5.8–6.0 ms（约 170 FPS），初始化约 450 ms，适合现场演示；CPU 约 86–91 ms 作为对照。

**观察与坑**

1. **运行页表头不可靠**：CPU 下选 FP16 或 FP32，表头都显示 `FP16`；GPU 下能正确区分 `GPU / FP16`、`GPU / FP32`；NNAPI、QNN GPU 正常显示后端名。所以 CPU 那三行实际是同一执行路径，86–91 ms 的差异属噪声，不能当作 FP16/FP32 的性能对比。
2. **INT8 档位与 FP32 模型不兼容**：报错
   `java.lang.IllegalArgumentException: Cannot copy to a TensorFlowLite tensor (serving_default_input_1:0) with 2880000 bytes from a Java Buffer with 720000 bytes.`
   即 INT8 档位按 uint8 分配输入（400×600×3 = 720,000 B），而模型输入是 float32（2,880,000 B）。**要在本机测 INT8，必须先量化出 INT8 模型**；另外 INT8 档位会额外放开 Qualcomm Hexagon / QNN DSP 选项（本机 FP16/FP32 下为灰色不可选）。
3. **选精度会重置后端**：切换 INT8/FP16/FP32 会把加速器重置为 CPU，必须“先选精度，再选后端”，否则实际跑的是 CPU（我曾因此得到一次表头为 `CPU / FP16` 的 91.0 ms 结果）。
4. **Arm NN 崩溃**：点 Run 后进程 abort 并重启，App 回到标准测试第 1 页，所有设置被清空。崩溃栈见 [`../data/armnn-crash.txt`](../data/armnn-crash.txt)：`armnn::ClImportTensorHandle::ClImport` 抛出未捕获的 C++ 异常（OpenCL 后端导入输入张量失败）。本机不采信 Arm NN 结果。
5. **模型选择会丢失**：崩溃后或切到别的页签再回来，`PATH TO .TFLITE FILE` 会变空，必须重新点 `Select Model`；该字段显示的是文档选择器 URI（`/document/msf:1000012122`），不是真实文件路径。
6. **NNAPI 比 CPU 还慢**（154 ms）：本机 NNAPI 很可能没有真正下沉到加速器，仅作记录，不作为结论。
7. 初始化和推理需要分开看：GPU 初始化 337–454 ms，单帧 5.8 ms；如果现场演示只跑 10 次，初始化占比会让“总耗时”看起来偏高，汇报时应说明这是**单帧推理延迟**。
