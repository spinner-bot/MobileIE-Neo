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

## 下一步

1. adb 授权通过后确认型号、系统、剩余空间；
2. `adb install` 安装 APK；
3. `adb push demo/org/demo-package/MobileIE-Demo/model.tflite /sdcard/Download/`；
4. 启动 App、截图，检查 Custom Model 入口。
