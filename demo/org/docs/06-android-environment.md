# 06 · zrh 个人安卓开发环境

- 建立日期：2026-11-13；作者：Codex  (sqs_A100)。
- 用户授权：安卓工具链安装到个人环境站 `/root/Users/zrh/.envs`，并在 `/root` 创建链接；只安装与验证环境，微 App 仍等待设计审阅。
- 实际目录：`/data/Users/zrh/.envs/android`；`/root/Users/zrh` 为既有兼容软链接。
- 便捷入口：`/root/android-dev` 指向上述实际目录。
- 安装状态：**已完成，E0–E2 工具、构建签名与模拟安装门控全部通过。** 本报告在 App 实施前提交；配置与原始证据见[环境记录](../results/android-environment-20261113/configuration.json)。

## 使用方式

```bash
source /root/android-dev/activate.sh
java -version
gradle --version
sdkmanager --list_installed
adb devices -l
# 启动个人模拟器（已安装镜像并建好 AVD）
bash demo/org/scripts/start-android-emulator.sh
# 仅关闭该模拟器
adb -s emulator-5580 emu kill
```

配置仅在 source 后对当前 shell 生效，不修改全局 Java、PATH 或其他用户环境。Android 用户状态、AVD、Gradle 依赖和临时缓存均设在个人环境站。独立 adb 使用 localhost:5038，避免占用默认 5037 上可能存在的其他用户服务；连接手机的本地电脑有自己的 adb 与授权。

## 组件与版本

| 组件 | 安装目标 |
| --- | --- |
| JDK | Eclipse Temurin 17.0.21+9，Linux x64 |
| Gradle | 8.13 |
| Android Gradle Plugin | 8.11.1，作为项目依赖预取及验证，不作为全局插件 |
| Kotlin Android 插件 | 2.2.10，作为项目依赖预取及验证 |
| SDK Command-line Tools | 19.0（13114758），`latest` 是指向该固定版本的本地链接 |
| SDK Platforms | Android 14 / API 34、Android 15 / API 35、Android 16 / API 36 |
| SDK Build Tools | 35.0.0、36.0.0；验证项目使用 35.0.0 |
| Platform Tools | 37.0.1-15733141；adb 1.0.41 |
| NDK | r27d，27.3.13750724 |
| CMake / Ninja | SDK CMake 3.22.1 及其附带 Ninja |
| Android Emulator | 37.2.13.0，build 16740729 |
| 系统镜像 | API 35 Google APIs x86_64，revision 9；已通过服务器 KVM 启动核验 |
| 预取推理依赖 | LiteRT CPU / GPU 1.4.2；只是依赖解析及打包检查，不代表模型/GPU 已验证 |

AGP 8.11.1 配套 Gradle 8.13、JDK 17，并支持 compileSdk 36，依据[官方兼容表](https://developer.android.com/build/releases/agp-8-11-0-release-notes)。平台版本与 App 的 minSdk 是不同概念，安装 API 36 不要求手机升级到 Android 16。

该服务器采用完整命令行工具链；可视化 Android Studio 可在本地电脑按需安装。系统镜像用于环境验证，拍照、相机回传与手机实际性能仍须在一加 9R 上独立检查。

## 目录布局

```text
/data/Users/zrh/.envs/android/
├── activate.sh
├── jdk/
├── gradle-8.13/
├── sdk/                   # 平台、构建工具、adb、NDK、CMake、模拟器、镜像
├── gradle-cache/
├── host-libs/             # 模拟器所需 Ubuntu 库的个人目录副本
├── bin/emulator           # 仅为模拟器启用上述宿主库的包装入口
├── android-user-home/     # AVD、adb 状态等用户资产
├── downloads/             # 下载包、源元数据
├── logs/                  # 安装和验证日志
├── records/               # 实际版本、摘要与核验结果
├── validation/            # 一次性工具链验证样例、产物及测试签名密钥
└── tmp/
```

环境、缓存、镜像、验证产物与密钥不提交 Git。版本化安装/核验脚本与文档保存在 `demo.org`，逐文件小步提交、分阶段推送。所有合并使用 `--no-ff`，禁止 rebase、squash、强推。

## 复现与核验范围

- 安装脚本：[install-android-environment.sh](../scripts/install-android-environment.sh)。JDK、Gradle 校验官方 SHA-256；命令行工具校验 SDK 元数据中的 SHA-1，并另外记录实际 SHA-256。SDK 包由 SDK Manager 校验并记录实际安装版本。授权安装过程中接受相关 SDK 组件许可证，记录保存在个人目录。
- 验证脚本：[validate-android-environment.sh](../scripts/validate-android-environment.sh)。在个人目录生成没有 Activity 的工具链探针，检查 Java/Kotlin、API 36 资源/字节码编译、ARM64 与 x86_64 C++ 交叉编译、Debug/Release APK、测试签名和 ZIP 对齐。
- 模拟器：独立 AVD、KVM、软件图形渲染；验证启动与 adb 包安装。该探针不包含微 App 的界面、拍照、baseline 或模型推理功能。
- 测试签名只用于验证构建链路，不能当作未来 App 正式签名密钥。

## 最终核验结果

| 检查 | 实际结果 | 证据 |
| --- | --- | --- |
| 目录、链接、缓存归属 | `/data` 已挂载，`/root/android-dev` 正确指向个人环境，adb 密钥也位于个人 Android 用户目录 | [配置清单](../results/android-environment-20261113/configuration.json) |
| Java 编译与执行 | JDK17 独立 Java 探针正常 | [Java 记录](../results/android-environment-20261113/java-probe.txt) |
| Kotlin、Debug/Release APK | 全部构建通过，2 分 20 秒，91 个任务执行；实际插件 Kotlin 2.2.10，Gradle 内置脚本 Kotlin 2.0.21 | [完整构建记录](../results/android-environment-20261113/validation-001.txt) |
| C++ 交叉编译 | ARM64 和 x86_64 Debug/Release 均通过，ARM64 ELF Machine 为 AArch64 | 同上 |
| 测试签名与对齐 | 签名验证通过（v3），16 KB ZIP 对齐检查通过 | 同上；完整对齐明细保存在个人环境 `records/zipalign.txt` |
| AVD 与 adb | `zrh-api35`，`emulator-5580`，KVM、软件图形渲染，启动完成属性为 1，已实际查看主屏截图 | [启动记录](../results/android-environment-20261113/emulator-start-001.txt) |
| 模拟器安装与包查询 | 测试签名探针安装 Success，`pm path dev.zrh.environmentprobe` 返回实际包路径 | [安装记录](../results/android-environment-20261113/probe-emulator-install.txt) |
| Python 数值参考环境 | 独立 venv，LiteRT 2.3.0 / NumPy 2.2.6 / Pillow 11.3.0；可加载官方 `[1,400,600,3]` 模型 | [依赖清单](../results/android-environment-20261113/reference-python-packages.txt)；尚未据此声明 App 数值通过 |

安装过程保留了以下问题与修复边界：

1. 模拟器最初缺少 `libpulse.so.0`。通过[宿主依赖脚本](../scripts/prepare-emulator-host-libs.sh)将 Ubuntu 软件包解包到 `host-libs/`，包装入口仅对模拟器设置库路径，没有全局安装；版本及摘要已入库。
2. 首轮 SDK 安装已完成组件下载，但末尾输出 `No AVD specified` 并返回 1；不将其记为成功。复核包清单、使用同一脚本重跑后正常返回 0，之后的完整编译、签名和模拟安装全部通过。[首轮日志](../results/android-environment-20261113/install-001.txt)、[正常重跑日志](../results/android-environment-20261113/install-002.txt)均保留；未把该输出未经定位地归因为具体 SDK 缺陷。
3. 构建有 Gradle 弃用提示，当前锁定的 Gradle 8.13 已正常通过，不据此宣称兼容 Gradle 9。

站点总占用约 10 GB（包含系统镜像、AVD、依赖缓存和探针产物，后续会变化）。精确安装包版本以入库清单为准；`platform-tools` 和 `emulator` 这种稳定通道包在未来重装时可能更新，应核对清单后再接受变更。

环境已具备 App 实施条件。当前通过的是无界面的工具链探针，不含 MobileIE 图片推理或拍照；这部分依据[07 交付门控](07-delivery-gates.md)继续实施和验收。真实一加 9R 尚未连接，不能从模拟器结果推导其功能或性能结论。
