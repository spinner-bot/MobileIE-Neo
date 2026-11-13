# 09 · 微光实验室 0.1.1 最终开发与模拟验收报告

日期：2026-11-13；作者：Codex  (sqs_A100)；分支：`demo.org`。

**后续更新：** 本文保留首轮 0.1.1 模拟交付时的结论。其后一加 9R 已完成主要功能检查，见[10 真机记录](10-real-device-0.1.1.md)；当前 Android 修复版为[0.1.2](11-android-0.1.2-state-fix.md)，Python 仍为 0.1.1。下列“尚未真机测试”描述的是本报告初次交付时状态。

**本轮自主目标已完成：个人安卓环境及配置报告、Android 微 App、Python 桌面伴随版、模拟测试与修正、安装包及验收证据均已交付。当前交付版本为 0.1.1。** 0.1.0 的发布文件与测试记录保留，不覆盖；完整实现背景和首轮迭代见[08 历史报告](08-app-delivery.md)。

本报告结论是开发与模拟验收通过。**新 App 尚未在真实一加 9R 安装运行，Python 版尚未在真实 Windows 运行，导师预检查尚未完成。** 11 月 12 日一加 9R 的 AI Benchmark 历史数据继续有效，但不作为新 App 的实测。最终验收仍定于 2026-11-16（周一）上午，导师提前检查，用户主要协助演示。

## 1. 环境已先行交付

个人环境实际目录 `/data/Users/zrh/.envs/android`，入口软链接 `/root/android-dev`。启用：

```bash
source /root/android-dev/activate.sh
```

JDK 17、Gradle、Android SDK/API 34/35/36、Build Tools、NDK、CMake/Ninja、adb、KVM 模拟器与 API 35 镜像已安装核验。Java/Kotlin、ARM64/x86_64 原生编译、Debug/Release、签名、对齐、模拟器启动和安装门控通过。个人缓存及私有 adb 5038 与其他用户隔离；未安装服务器 GUI Android Studio 或 Windows 原生开发工具链。

[环境配置报告](06-android-environment.md)于 `3ef90b6c` 提交并推送，早于首次 App 源码提交 `311c1b54`，满足先环境、报告提交、后开发的顺序。

## 2. 交付文件

固定目录：`/data/Users/zrh/data/MobileIE-Neo/demo/org/releases/0.1.1/`。

| 文件 | 用途 | 大小 |
| --- | --- | --- |
| `MobileIE-LightLab-0.1.1.apk` | Android 10+ 声明，ARM64/x86_64，开发调试签名；已测 Android 15 模拟器 | 18,115,838 B |
| `MobileIE-LightLab-Android-Demo-0.1.1.zip` | APK、中文手机说明、模拟操作录像、摘要清单 | 9,556,024 B |
| `MobileIE-LightLab-Desktop-Python-0.1.1.zip` | Python 程序、依赖清单、Windows 启动器、同一模型及三张样例 | 976,552 B |
| `MobileIE-LightLab-0.1.1-emulator-demo.mp4` | 28.837 秒、720×1560 的最终模拟操作录像 | 2,424,192 B |

APK SHA-256：`3451182678596d3672d3a7e81c5b12cd6ca609cb50695665e7b44e51234447a8`。

全部路径与 SHA-256 见[发布清单](../results/app-0.1.1/release.json)。两次打包字节一致，两个 ZIP 的 CRC 和所有交付文件摘要均通过；已从桌面 ZIP 实际解压运行模型，确认便携资源路径正确、样例原始输出及 PNG 与官方参考完全一致，见[包核验](../results/app-0.1.1/package-verification.json)。大文件按仓库资产约定保存在个人项目数据站；源码、报告、截图、原始日志和摘要入 Git。

## 3. 功能与视觉交互

Android 支持系统相机完整照片回传、系统选图、三张内置暗光样例、同坐标滑动对比、MobileIE/Gamma 切换、三图总览、PNG/三联图/JSON 保存与 AI Benchmark 主入口启动。照片先校正方向并统一到 600×400 居中 3:2 输入，三种画面共享取景；这是拍照后单张处理。

传统 baseline 固定为逐 RGB 通道 `y=x^0.5`，不含降噪。官方模型不改动，摘要为 `2ea09b840634469eceab5263bf7b549c33fcd5df2a25701ce5fefc585402ad44`。Android 使用 CPU 内置算子四线程，桌面沿用官方默认单线程；App 的计时与 AI Benchmark 各自独立。AI Benchmark 仍需手动选择 Custom Model 和后端。

两端采用暖白底色、深绿信息区与陶土色主按钮。主流程突出获取照片、处理、比较、保存；技术记录放入二级入口。计算时显示进度并禁用重复操作，模型失败时明确标识“未生成”，Gamma 独立可用，保存中途失败会回滚。桌面采用 Tkinter，支持窗口缩放与侧栏滚动，使用已有 Python，无需 Windows 原生开发工具。

已实际查看以下原始截图，未生成或修饰截图：

- Android：[首屏](../results/app-0.1.1/screenshots/01-fresh-home.png)、[滑动对比](../results/app-0.1.1/screenshots/11-reviewed-result.png)、[三图总览](../results/app-0.1.1/screenshots/12-reviewed-overview.png)、[失败状态](../results/app-0.1.1/screenshots/06-model-failure-honest-baseline.png)、[1.3 倍字体](../results/app-0.1.1/screenshots/09-font-scale-1.3.png)。
- Python：[首屏](../results/desktop-0.1.1/desktop-01-home.png)、[对比](../results/desktop-0.1.1/desktop-02-result.png)、[总览](../results/desktop-0.1.1/desktop-03-overview.png)、[小窗口滚动](../results/desktop-0.1.1/desktop-06-small-window-scrolled.png)。

## 4. 最终质量门控

| 项目 | 实际结果与证据 |
| --- | --- |
| Android 完整回归 | **14 项，0 失败，90.665 秒**；Gamma 全 256 值、8 种方向、共同输入、真实系统相机与选图、取消/坏图/空相机、双击、重建、对比、保存回读、模型失败与导出回滚均通过。[仪器日志](../results/app-0.1.1/instrumentation.txt) |
| Android 数值 | 三样例、黑、白、RGB 梯度共 **6 组**；对未改动官方脚本最大 raw 差 **2.384185791015625e-7**、uint8 最大差 **1**，全有限，全部通过原定 `1e-4 / 1` 门槛。[完整数值表](../results/app-0.1.1/numerical-report.json) |
| 重复处理状态 | 在同一主线程回调内强制断言忙碌状态，无条件跳过路径：方法与对比区禁用、计时入口禁用、等待文案及无障碍提示正确、重复提交只增加一次运行编号。[验收摘要](../results/app-0.1.1/qa-summary.json) |
| 保存回读 | 强化回读已纳入本次完整套件：成功方法 PNG 的 RGB 完全相同、三联图 1800×478 且内容正确、JSON 正确、MediaStore 无待发布条目；失败回滚通过。 |
| 生命周期 | 冷启动恢复相同会话与两种结果；外部相机前台结束本 App 进程，返回新照片仍可处理。[冷启动](../results/app-0.1.1/process-restore.json)、[相机进程恢复](../results/app-0.1.1/camera-process-recovery.json) |
| 外部入口 | 缺失分支通过；存在分支使用明确标识的临时桩包通过，随后已卸载。**不等于真实 AI Benchmark 联调。**[分支记录](../results/app-0.1.1/benchmark-installed-branch.json) |
| 构建与签名 | Lint **0 错误、6 条固定依赖有新版提示**；签名、16 KB ZIP 对齐、APK 内模型摘要通过。[Lint](../results/app-0.1.1/lint-results-debug.txt)、[签名](../results/app-0.1.1/signature.txt) |
| Python 核心及 GUI | Linux/Xvfb 下 **5 项全部执行通过，无跳过，7.232 秒**；完整交互、小窗口、异常、PNG/JSON、退出清理通过。[日志](../results/desktop-0.1.1/desktop-validation.txt) |
| Python 数值 | 三张样例 raw 最大差 **0**，PNG 精确一致。[数值表](../results/desktop-0.1.1/desktop-numerics.json)；不是 Windows 实测。 |
| 录像 | 追加 **1 项实际解码测试通过**，28.837 秒，中间帧非空且已实际查看。[解码结果](../results/app-0.1.1/video-decoder.json)、[实际帧](../results/app-0.1.1/screenshots/video-decoded-midpoint.png) |

环境 E0–E3 与 Android A0–A5 的结束条件见[门控定义](07-delivery-gates.md)。数值检查验证部署一致性，不证明模型对所有照片的画质优势；模拟相机是合成场景，模拟器耗时不能当作手机测速。

## 5. 本次收口修正

0.1.0 打包复核时发现重新处理仍可切换旧输出，计时区域有机会读取未完成元数据。0.1.1 增加忙碌保护、等待文案和无障碍提示，并补强不可跳过的断言。Android 全套重新运行；桌面仅同步版本字段，仍重跑全部 5 项。未增加功能范围，未放宽数值门槛。

最终录像首轮因停止后 MP4 的 `moov` 尚未落盘而失败，已[保留失败日志](../results/app-0.1.1/recording-attempt-001.txt)。录制脚本现在等待容器完成写入；重录后另用 Android 媒体解码器验证。0.1.0 的首轮编译、Lint、UI 与桌面失败及修正记录继续保留在历史结果目录。

## 6. 复现与现场接续

```bash
bash demo/org/scripts/start-android-emulator.sh
bash demo/org/scripts/run-android-app-gates.sh
bash demo/org/scripts/run-desktop-validation.sh
source /root/android-dev/activate.sh
"$ANDROID_DEV_ROOT/validation/reference-env/bin/python" -I demo/org/scripts/verify-android-lifecycle.py
```

开发入口见[Android README](../app/README.md)、[Python README](../desktop/README.md)；手机安装与操作见[中文说明](../demo-package/微光实验室操作说明.txt)。全部演示改动在独立 `demo.org` 工作树小步提交、阶段推送；不修改 `main` 研究工作，所有合并要求显式 `--no-ff`，未 rebase、squash 或强推。

剩余设备及人员环节已列入[预检查清单](04-acceptance-checklist.md)：

- [ ] 一加 9R 安装新 APK，真实拍照、选图、比较、保存、真实 AI Benchmark 跳转与返回，并记录实际 CPU 耗时。
- [ ] Windows 本机检查 Tcl/Tk、DLL、中文字体、DPI、系统文件对话框与真实文件保存。
- [ ] 导师预检查，确认最终展示方式、时间、地点与连接，保存实际反馈和真机录屏。

上述环节需要设备或人员接续，未伪报完成；本轮已授权的环境建设、软件实现、模拟验收、优化和交付具有完整结果及证据，达到本轮自主目标结束条件。
