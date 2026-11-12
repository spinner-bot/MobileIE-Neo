# 微光实验室 · MobileIE Android 演示 App

原生 Java Android App，离线运行官方 `lolv1.tflite`，提供现场拍照、相册选图、三张内置暗光样例，以及原图 / MobileIE / Gamma 三方对比。输入统一校正方向、居中裁剪到 3:2 并缩放为 600×400。Gamma 固定使用逐 RGB 通道 `y=x^0.5`，不是降噪方法。

## 构建与安装

从仓库根目录执行：

```bash
source /root/android-dev/activate.sh
bash demo/org/app/build.sh assembleDebug
adb -s emulator-5580 install -r demo/org/app/app/build/outputs/apk/debug/app-debug.apk
adb -s emulator-5580 shell am start -n dev.zrh.mobileie/.MainActivity
```

个人工具链配置见[环境报告](../docs/06-android-environment.md)。Gradle、AGP、Java 及依赖版本随配置锁定；应用使用 CPU 内置算子与 4 线程，不依赖 Google Play 服务。最低声明 Android 10；实际已测试平台以验收报告为准，不能把最低 API 声明当作所有设备都已验证。

## 现场操作

1. 点击“内置样例”先熟悉流程；现场也可“拍照”或“相册选图”。拍照会打开系统相机，拍完确认后回到 App。
2. 检查“原图（处理范围）”的居中裁剪，再点击“开始对比”。本 App 单张处理，不提供摄像头实时增强取景。
3. 使用 MobileIE / Gamma 0.5 切换大图对比，拖动分界线查看同一位置；“三图总览”便于整体讲解。
4. 向下滚动可查看耗时、保存结果或打开 AI Benchmark。点击耗时可查看分项记录；不要将模拟器或本 App 的计时冒充 AI Benchmark 实测。
5. 保存后，相册 `Pictures/MobileIE` 中包含输入、成功生成的方法结果和带标签对比图；`Download/MobileIE` 保存 JSON 记录。未生成的方法明确标识，不伪造结果。

App 没有联网权限，图片在本机处理。相册使用系统文档选择器；相机使用 FileProvider 全尺寸输出，不需要直接访问摄像头的权限。超过 32 MB 或不可解码的输入会提示重新选择。取消操作保留上次结果；更换图片后需重新处理。

## 与 AI Benchmark 配合

App 内模型和既有 `model.tflite` 字节一致，摘要见 [asset-sources.json](asset-sources.json)；App 的“关于模型与演示”也可查看摘要。AI Benchmark 负责独立测速，微光实验室展示真实图片处理。按钮只尝试打开其主入口，仍需在 AI Benchmark 中手动进入 PRO Mode → Custom Model、选择模型与后端。

未安装 AI Benchmark 时有明确提示，仍能使用图片对比。首版没有外部 App 推理委托、自动传图或自动读取跑分功能。

## 核验入口

```bash
bash demo/org/scripts/start-android-emulator.sh
bash demo/org/scripts/run-android-app-gates.sh
source /root/android-dev/activate.sh
"$ANDROID_DEV_ROOT/validation/reference-env/bin/python" -I demo/org/scripts/verify-android-lifecycle.py
```

第一项 App 门控包含 14 个仪器测试、官方脚本独立数值核对、Lint、签名和对齐检查。第二项在系统相机前台时结束本 App 进程，验证返回的照片仍可被接收并处理。测试针对个人 AVD，会生成模拟照片与 MediaStore 文件；不要直接把这些故障注入测试用于装有个人照片的正式使用设备。

## 主要源码与范围

| 模块 | 文件 |
| --- | --- |
| 中文界面、系统相机与选图入口 | `app/src/main/java/dev/zrh/mobileie/MainActivity.java` |
| 共同图片预处理与 Gamma | `ImagePipeline.java` |
| 官方模型校验与 CPU 推理 | `MobileEngine.java` |
| 任务、后台线程与持久化恢复 | `LightLabApplication.java` |
| 同位置绘制、滑动及无障碍操作 | `ComparisonView.java` |
| 带标签导出与部分写入回滚 | `ResultExporter.java` |

源码、样例来源和 MobileIE Apache-2.0 许可随仓库保存。APK 及较大的运行/测试产物存放 zrh 自有项目数据目录，最终位置、摘要、功能与未验证范围见[交付报告](../docs/08-app-delivery.md)。演示安装包采用开发调试签名，不作为应用商店正式发布包。
