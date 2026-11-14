# 08 · 微光实验室 0.1.0 交付与模拟验收报告

日期：2026-11-13；作者：Codex  (sqs_A100)；分支：`demo/org`。

**历史版本说明：双端 0.1.1 交付见[09 报告](09-app-v011-final.md)，当前 Android 0.1.2 修复版见[11](11-android-0.1.2-state-fix.md)。** 本文及下列 0.1.0 结果保留为首轮验收与迭代记录，不作为当前安装包摘要。

**安卓开发环境、Android App、Python 桌面对比工具、模拟测试与优化迭代已完成。** 本报告对应源码、安装包、原始日志、失败记录、实际截图和摘要；验收安排仍为 2026-11-16（周一）上午，导师提前检查，用户协助演示。

这里完成的是开发与模拟验收。**新 App 尚未在真实一加 9R 安装测试，Python 版尚未在真实 Windows 机器运行，导师预检查也尚未进行。** 历史 9R AI Benchmark 测速保留原结论，不冒充新 App 的实测。

## 1. 交付文件

大体量安装包与视频存放 zrh 自有项目资产目录，Git 保存源码、完整门控证据、截图、来源与摘要。交付目录为：

`/data/Users/zrh/data/MobileIE-Neo/demo/org/releases/0.1.0/`

| 文件 | 用途 | 大小 |
| --- | --- | --- |
| `MobileIE-LightLab-0.1.0.apk` | Android 演示安装包，ARM64 / x86_64，开发调试签名 | 24,101,050 B |
| `MobileIE-LightLab-Android-Demo-0.1.0.zip` | APK + 中文手机说明 + 操作录屏 + SHA256SUMS | 9,487,371 B |
| `MobileIE-LightLab-Desktop-Python-0.1.0.zip` | Python 程序、运行依赖清单、Windows 启动器、模型与样例 | 976,551 B |
| `MobileIE-LightLab-0.1.0-emulator-demo.mp4` | 27.524 秒模拟器操作录屏，720×1560 | 2,345,509 B |

全部路径、精确 SHA-256 和版本字段见[发布清单](../results/app-0.1.0/release.json)。APK SHA-256 为：

`333028916ca87aab4eae129a49877f29806eb2f810f76f776dc9c0c7d21f25a6`

已逐文件复核交付摘要、两个 ZIP 的 CRC，并实际从桌面 ZIP 解压后加载模型处理样例，确认 `assets/` 布局可用，输出与参考 PNG 逐像素相同，见[交付包核验](../results/app-0.1.0/package-verification.json)。录屏已由 Android 媒体解码器解出中间帧并检查非空画面，见[视频门控](../results/app-0.1.0/video-decoder.json)。

## 2. 环境与实现

环境先完成并提交报告（`3ef90b6c`），随后开始 App 实施。[06 环境报告](06-android-environment.md)记录 JDK 17.0.21+9、Gradle 8.13、AGP 8.11.1、API 34/35/36、Build Tools 35/36、NDK r27d、CMake 3.22.1、adb、模拟器及构建/签名/安装结果。

个人环境实际位置 `/data/Users/zrh/.envs/android`，`/root/android-dev` 为软链接。`source /root/android-dev/activate.sh` 后启用本套工具；缓存与 Android 用户状态属于个人目录，adb 使用私有 localhost:5038。未在服务器配置 GUI Android Studio，也未安装 Windows 原生开发工具链。

| 能力 | Android App | Python 桌面版 |
| --- | --- | --- |
| 图片输入 | 系统相机完整照片、系统选图、三张离线样例 | 选图、内置样例 |
| 共同处理范围 | 校正方向、sRGB/RGB、居中 3:2 裁剪、600×400 | 同一处理规则；平台缩放实现不宣称对任意照片逐像素一致 |
| 官方模型 | LiteRT 1.4.2，CPU 内置算子，4 线程 | ai-edge-litert 2.3.0，关闭默认 delegate，官方默认单线程 |
| 传统对照 | 固定逐 RGB 通道 `y=x^0.5`，不含降噪 | 相同公式 |
| 对比 | 同位置滑动、Gamma 切换、三图总览 | 相同交互目标，窗口可缩放，侧栏可滚动 |
| 保存 | MediaStore PNG / 三联图 / JSON，部分失败回滚 | 新目录 PNG / 三联图 / JSON，部分失败回滚 |
| AI Benchmark | 尝试打开已安装包；缺失时提示 | 继续由 Android 手机上的 AI Benchmark 独立测速 |

同一份官方模型大小 23,228 B、float32、`[1,400,600,3]`，APK 内模型与原模型摘要一致。样例与许可来源见[逐文件来源清单](../app/asset-sources.json)。

## 3. 质量门控实绩

| 门控 | 结果 | 主要证据 |
| --- | --- | --- |
| 环境 E0–E3 | Java 编译执行、Kotlin、两 ABI 原生编译、Debug/Release、签名、对齐、KVM 启动及包安装均通过，配置报告已提交 | [环境配置与日志](../results/android-environment-20261113/configuration.json) |
| Android A0–A3 | 完整 **14 项仪器测试，0 失败**，91.375 秒；包含 Gamma 256 值、8 种方向、共同输入、坏图、真实系统相机回传与取消、系统选图、滑动、保存、Activity 重建、双击防重入、模型失败与导出回滚 | [完整仪器日志](../results/app-0.1.0/instrumentation.txt) |
| Android A1 数值 | 三张样例 + 黑图 + 白图 + RGB 梯度，共 **6 组**；对未修改官方脚本最大原始误差 **2.384185791015625e-7**，uint8 最大差 **1**，全有限，全部通过原门槛 `1e-4 / 1` | [全数值表](../results/app-0.1.0/numerical-report.json) |
| Android 导出专项 | 追加定向测试通过：三种 PNG 回读 RGB 完全相同、三联图尺寸与内容正确、JSON 字段正确、MediaStore 文件已发布 | [回读专项](../results/app-0.1.0/app-export-roundtrip-001.txt) |
| Android 进程恢复 | 冷启动恢复原会话与结果；相机前台期间结束本 App 进程，返回后产生新会话并完成两种方法计算 | [普通恢复](../results/app-0.1.0/process-restore.json)、[相机进程恢复](../results/app-0.1.0/camera-process-recovery.json) |
| 外部启动分支 | 缺失分支由完整 UI 测试验证；存在分支用明确标识的临时桩包验证成功，随后卸载。**并非真实 AI Benchmark 联调** | [启动分支记录](../results/app-0.1.0/benchmark-installed-branch.json) |
| Android A4 视觉 | 实际查看首启、完成、总览、模拟拍照、失败、恢复、1.3 倍字体截图，修正无效选项与缺失输出标识 | [实际审阅记录](../results/app-0.1.0/visual-review.json) |
| Android A5 构建交付 | Lint **0 错误、6 条固定依赖有新版的提示**；签名与 ZIP 对齐通过；打包与源模型摘要通过 | [Lint](../results/app-0.1.0/lint-results-debug.txt)、[签名](../results/app-0.1.0/signature.txt)、[发布清单](../results/app-0.1.0/release.json) |
| Python 桌面核心与 GUI | **5 项测试全部执行通过，无跳过**；含模型、Gamma、输入、保存回滚、GUI 完整流程与小窗口滚动、关闭清理 | [最终模拟核验日志](../results/desktop-0.1.0/desktop-validation-wrapper-001.txt) |
| Python 数值 | 本次 Linux 测试三张样例 raw 最大差 **0**，PNG 逐像素相同；不是 Windows 实测结论 | [桌面数值表](../results/desktop-0.1.0/desktop-numerics.json) |
| 录屏 | MP4 CRC/摘要、真实解码、720×1560、27.524 秒，中间帧非空且已实际查看 | [视频测试](../results/app-0.1.0/video-test-001.txt)、[解码帧](../results/app-0.1.0/screenshots/video-decoded-midpoint.png) |

Android 的 14 项为主套件；导出回读强化与视频解码属于追加专项，不把重复执行计为新的独立样本。数值核验验证部署链路一致性，不证明 MobileIE 相比 Gamma 的画质优势。模拟器相机是合成场景，时间也不是手机实测。

## 4. 视觉与交互

两端采用浅暖底色、深绿文字与陶土色主按钮。流程围绕获取照片、确认范围、处理、对比、保存展开；计算时禁用重复提交并显示进度，失败时保留可解释的状态。底层记录放在二级信息入口。

Android 的滑动与三图总览共享相同图像坐标，方法切换不改变取景；没有模型结果时禁用该选项，并明确显示“未生成”。桌面总览标签采用界面字体绘制，避免随图片缩小而不可读；小窗口通过侧栏滚动访问完整说明。

- [安卓首屏](../results/app-0.1.0/screenshots/01-fresh-home.png)、[安卓处理对比](../results/app-0.1.0/screenshots/11-reviewed-result.png)、[安卓三图总览](../results/app-0.1.0/screenshots/03-three-method-overview.png)、[安卓失败状态](../results/app-0.1.0/screenshots/06-model-failure-honest-baseline.png)。
- [桌面首屏](../results/desktop-0.1.0/desktop-01-home.png)、[桌面对比](../results/desktop-0.1.0/desktop-02-result.png)、[桌面总览](../results/desktop-0.1.0/desktop-03-overview.png)、[小窗口滚动](../results/desktop-0.1.0/desktop-06-small-window-scrolled.png)。

## 5. 已修正问题与保留边界

1. 首轮 Android 编译的 `Files.readString` 不在 Android API 中，改为兼容的字节读取；Lint 发现的 API 30 边距调用改为 AndroidX 兼容实现，保留 Android 10 声明。
2. 系统文档选择器的首轮自动化命中了“预览”按钮；改用列表和精确选择，实际系统选图流程已通过。没有把这次失败删去或提前计为成功。
3. 补强恢复期接收照片、空相机输出、旧计时清除与模型失败状态；相机进程死亡路径另行实际检查。
4. 拖动绘制对象改为复用；模型失败时无效选项禁用，空白图块明确标识。没有通过隐藏 Lint 错误或放宽数值门槛收口。
5. Python 四线程最初相对官方默认单线程出现少量 1 像素差；确认输入完全相同后，采用官方单线程路径，保持原有 PNG 精确一致要求。Linux 虚拟显示生命周期及小窗口裁切已修正。

首轮编译、Lint、UI 测试与桌面失败日志均保留在对应结果目录。导出回滚使用受控中途失败验证，不把它写成真实磁盘耗尽测试。真实 AI Benchmark 只在既有 9R 历史工作中运行过，本次新增 App 的已安装启动分支使用测试桩，不能据此声称完成真实第三方 App 联调。

## 6. 使用与真机接续

- 手机安装与演示：[微光实验室操作说明](../demo-package/微光实验室操作说明.txt)。
- Android 源码、构建与核验：[App README](../app/README.md)。
- Windows Python 运行：[桌面 README](../desktop/README.md)，使用已有演示 Python，无需 Windows 原生开发工具链。

接到设备后仍需：

- [x] 一加 9R 安装 APK，真实暗光拍照、相册选择、比较与保存完整走一遍，记录本 App 实际 CPU 耗时。→ 2026-11-13 完成，见[10 真机检查](10-real-device-0.1.1.md)（拍照推理 106.7 ms、总处理 200.6 ms）。
- [x] 在已安装真实 AI Benchmark 的手机上检查按钮跳转与返回，再按既有 GPU FP16 流程展示独立测速。→ 跳转已通；返回需经系统切换，后续现场测速仍按既有 GPU FP16 流程。
- [~] Windows 本机检查 Python/Tcl/Tk、DLL、原生文件对话框、DPI、字体和真实文件保存。→ 2026-11-13 用户决定：本工作属移动端演示，Windows 桌面版为“备用中的备用”，移动端跑通后不再列为验收项；产物与模拟记录保留归档，不建议现场使用。
- [ ] 导师提前检查并确认最终演示形式、时间、地点与连接方式；保留实际反馈及新录屏。

上述真机与导师环节有明确标注，未伪报完成。本轮可自主完成的环境、实现、模拟验收、迭代、交付与版本化证据已具备结束条件；后续真机检查按实际设备状态接续。
