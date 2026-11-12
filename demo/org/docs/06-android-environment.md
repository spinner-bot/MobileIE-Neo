# 06 · zrh 个人安卓开发环境

- 建立日期：2026-11-13；作者：Codex  (sqs_A100)。
- 用户授权：安卓工具链安装到个人环境站 `/root/Users/zrh/.envs`，并在 `/root` 创建链接；只安装与验证环境，微 App 仍等待设计审阅。
- 实际目录：`/data/Users/zrh/.envs/android`；`/root/Users/zrh` 为既有兼容软链接。
- 便捷入口：`/root/android-dev` 指向上述实际目录。
- 安装状态：进行中，最终结果在核验完成后补充；本文列出的组件是安装目标。

## 使用方式

```bash
source /root/android-dev/activate.sh
java -version
gradle --version
sdkmanager --list_installed
adb devices -l
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
| Platform Tools | SDK 仓库安装时的稳定版本，最终清单记录实际版本 |
| NDK | r27d，27.3.13750724 |
| CMake / Ninja | SDK CMake 3.22.1 及其附带 Ninja |
| Android Emulator | SDK 仓库安装时的稳定版本，最终清单记录实际版本 |
| 系统镜像 | API 35 Google APIs x86_64；用于服务器 KVM 模拟器核验 |
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

待实际安装和验证正常结束后填写。未完成项不得记录为通过。
