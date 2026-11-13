# 11 · Android 0.1.2 样例状态刷新修复

日期：2026-11-13；作者：Codex  (sqs_A100)；分支：`demo.org`。

已 fetch，并以 **`--no-ff` 合并提交 `228d4dd6`** 接入本地的一加 9R [0.1.1 真机记录](10-real-device-0.1.1.md)。针对其中“连续选样例后仍停留在初始界面”的隐患，完成受控复现、最小修正与完整模拟回归，交付 **Android 0.1.2 / versionCode 3**。

**0.1.2 尚未在一加 9R 实测。** 本次复现确认的是可导致相同症状的页面回调缺陷；原真机当时经过截图编辑器等操作的完整触发链仍需同机复验，不能据此声称其所有偶发路径都已消除。Python 桌面版保持 0.1.1。

## 1. 复现与修正

旧实现只在 `onCreate` 注册一个进程级界面观察器。第二个同类 Activity 接管观察器后退出，旧 Activity 恢复时不会重新注册，后台加载会继续成功，但旧页面收不到结果，因而仍显示“选一张暗光照片”，“开始对比”保持禁用。

在已发布 **0.1.1 原 APK** 上执行实际回退栈测试：启动两层 MainActivity → 关闭上层 → 确认原实例恢复 → 通过样例对话框选择书柜。测试确认新会话已生成，随后因页面不刷新而失败：`Sample loaded, but visible Activity did not refresh`。被测原 APK SHA-256 与既有发布摘要完全一致：`3451182678596d3672d3a7e81c5b12cd6ca609cb50695665e7b44e51234447a8`。

修正内容：

- `MainActivity.onResume` 重新绑定观察器，并立即显示当前状态；`onPause` 解绑。原有按观察器身份解绑的保护保留，迟到的销毁不会清除当前前台页面的回调。
- 升级后重新处理旧会话时，JSON 的 `app_version` 更新为实际运行版本；尚未重新计算的旧结果仍保留原记录。
- 模型、Gamma、CPU 线程数、图片预处理、视觉布局和计时定义均未调整。

[旧包预期失败](../results/app-0.1.2/regression-red-0.1.1.txt)、[修正后三项定向通过](../results/app-0.1.2/regression-green-0.1.2.txt)。首次测试夹具曾将 ActivityScenario 跟踪的另一个实例误认成原实例，已改用生命周期监视器确认实际对象；[夹具失败日志](../results/app-0.1.2/test-harness-attempt-001.txt)单独保留，不作为产品复现证据。

## 2. 最终验收结果

| 门控 | 结果与证据 |
| --- | --- |
| 完整仪器回归 | **17 项、0 失败，112.548 秒**；原 14 项加返回旧页面、后台完成后恢复、连续 6 轮回退/重建三项。[最终日志](../results/app-0.1.2/instrumentation.txt) |
| 升级旧会话 | 完整套件包含旧版本记录重新处理后版本正确的断言。[摘要](../results/app-0.1.2/qa-summary.json) |
| 数值 | **6 组全部通过**，最大 raw 差 **2.384185791015625e-7**，uint8 最大差 **1**，未放宽原门槛。[数值表](../results/app-0.1.2/numerical-report.json) |
| 相机进程恢复 | 相机前台结束 App 进程，返回的新照片仍能接收并完成两种处理。[记录](../results/app-0.1.2/camera-process-recovery.json) |
| 外部入口 | 已安装入口用明确标识的桩包验证通过，随后卸载；不是本次真实 AI Benchmark 联调。[记录](../results/app-0.1.2/benchmark-installed-branch.json) |
| 构建与升级 | Lint **0 错误、6 条固定依赖提示**；签名与 ZIP 对齐通过；与 0.1.1 的签名证书相同，支持保留数据覆盖升级。[证书](../results/app-0.1.2/signing-compatibility.txt) |
| 交付 | APK 摘要、ZIP CRC 与 ZIP 内 APK 摘要通过。[清单](../results/app-0.1.2/release.json)、[包核验](../results/app-0.1.2/package-verification.json) |

三项定向回归也在最终 17 项中执行，不将重复运行累计成新的独立测试数。已实际查看新版[完成态](../results/app-0.1.2/screenshots/02-mobileie-comparison.png)与[失败态](../results/app-0.1.2/screenshots/06-model-failure-honest-baseline.png)截图，布局及状态标签正常。

## 3. 修复包与安装

目录：`/data/Users/zrh/data/MobileIE-Neo/demo/org/releases/0.1.2/`。

- `MobileIE-LightLab-0.1.2.apk`，18,116,002 B。
- `MobileIE-LightLab-Android-Fix-0.1.2.zip`，7,288,319 B；含上述 APK、中文手机说明、更新说明和 SHA256SUMS，不含新录像或桌面程序。

APK SHA-256：`5484907db7c8e87c4a270a24ab8625471f67f4d68edb1229810b738899ec30c0`。

```bat
adb install -r MobileIE-LightLab-0.1.2.apk
```

无需卸载或清数据。0.1.0/0.1.1 发布文件与真机记录保留；Python ZIP 与流程参考录像仍使用既有 0.1.1。[手机说明](../demo-package/微光实验室操作说明.txt)已更新。

开发复现入口：

```bash
source /root/android-dev/activate.sh
bash demo/org/scripts/start-android-emulator.sh
bash demo/org/scripts/run-android-app-gates.sh
"$ANDROID_DEV_ROOT/validation/reference-env/bin/python" -I demo/org/scripts/verify-android-lifecycle.py
```

## 4. 一加 9R 接续检查

- [ ] 覆盖安装，确认 `versionName=0.1.2` / `versionCode=3`。
- [ ] 返回上次会话，重新处理并保存，确认本次记录版本为 0.1.2。
- [ ] 相机拍摄/取消、相册、真实 AI Benchmark 往返、系统截图编辑器返回后，各连续选择样例并点击处理，确认状态与按钮持续可用。
- [ ] Home 后台、重复进入页面、最近任务切换后再选样例；留存本次截图、日志与结果，不覆盖 0.1.1 原记录。

Windows 实机与导师预检查仍待接续。本轮服务器端修复、回归、交付及分阶段推送完成；真机复验单独保留未完成状态。
