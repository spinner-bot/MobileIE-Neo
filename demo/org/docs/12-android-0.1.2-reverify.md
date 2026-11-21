# 12 · 安卓 0.1.2 一加 9R 真机复验

- 执行：2026-11-13 10:09–10:15 · claude code (local)，Windows 11 + adb 37.0.1（USB）
- 对象：**微光实验室 0.1.2**（`dev.zrh.mobileie`，versionCode 3）
- 关联：[11 修复报告](11-android-0.1.2-state-fix.md)、[10 0.1.1 真机检查](10-real-device-0.1.1.md)、证据 [`results/phone-0.1.2/`](../results/phone-0.1.2)
- 结论：**覆盖升级成功、数据保留；报告所述回调缺陷的真机触发链（第二层实例关闭后返回旧实例）与多轮往返 + 连续选样例均通过**。导师预检查仍未做；Windows 桌面版按用户 2026-11-13 决定不再作为验收项。

## 1. 升级

| 项 | 值 |
| --- | --- |
| APK | `MobileIE-LightLab-0.1.2.apk`，18,116,002 B |
| SHA256 | `5484907db7c8e87c4a270a24ab8625471f67f4d68edb1229810b738899ec30c0`（与 11 报告一致） |
| 安装 | `adb install -r` 返回 `Success`，未卸载、未清数据 |
| 版本 | `versionName=0.1.2`、`versionCode=3`；`firstInstallTime` 仍为 0.1.1 的 2026-11-13 09:22:42 |
| 旧数据 | `Download/MobileIE/` 中 0.1.1 记录保留，`app_version` 仍为 `0.1.1`（未重新处理，故未改写） |

升级后首屏提示「已恢复上次照片，可继续查看或重新处理」，恢复的是上次未完成的样例 1 会话，属于预期行为。

## 2. 复验过程与结果

全部为真机操作，每步以 UI 树文本核对状态，不只依赖屏幕观感。

| 步骤 | 结果 |
| --- | --- |
| 冷启动 | 638 ms（升级后首次，LaunchState: COLD） |
| **报告根因链**：起第二层 `MainActivity` → 返回键关闭上层 → 回到旧实例 → 选「01 · 书柜」 | **通过**：状态变为样例 1 / 「已恢复上次照片」，`开始对比` 可用 |
| 往返 1：现场拍照 → 返回键取消 | 回到本 App；随后选「02 · 暗光样例」→ **立即刷新为样例 2** |
| 往返 2：相册选图 → 系统选择器 → 返回键取消 | 回到本 App；随后选「01 · 书柜」→ **立即刷新为样例 1** |
| 往返 3：AI Benchmark（`org.benchmark.demo/.RunClassifyModel`）→ 切回本 App | 随后选「03 · 仓库样例」→ **立即刷新为样例 3** |
| 连续选择：01 → 02 → 03（不离开页面） | 三次均在约 4 s 内刷新，状态与样例编号逐一对应 |
| 完整处理（样例 3） | 「处理完成 · 两种方法均在本机计算」 |
| 保存 | 提示已存相册与记录；新记录 `app_version=0.1.2`，旧 0.1.1 文件不受影响 |

证据：[恢复会话](../results/phone-0.1.2/01-restored-session.png)、[往返后样例刷新](../results/phone-0.1.2/02-sample-refresh-after-roundtrips.png)、[处理结果](../results/phone-0.1.2/03-result.png)、[保存后（含 v0.1.2 页脚）](../results/phone-0.1.2/04-saved-gamma.png)。

### 2.1 本次计算记录（样例 3）

[`record-sample3.json`](../results/phone-0.1.2/record-sample3.json)：`app_version=0.1.2`，`inference_ms=124.35`，`packing_ms=144.41`，`rendering_ms=42.45`，`processing_total_ms=322.12`，`gamma_ms=10.81`，`model_sha256` 与发布一致，runtime 仍为 `LiteRT 1.4.2 CPU built-in, 4 threads`。

本次数值高于 0.1.1 的实拍记录（106.7 ms），发生在熄屏唤醒、多次往返之后，属单次样本；不作为性能回归结论。计时口径未变。

## 3. 判定

- 11 报告所述「旧 Activity 恢复后收不到加载结果」的缺陷在真机上**已消除**；
- 但 0.1.1 期间观察到的失效还夹杂系统截图编辑器覆盖前台等操作，本次已覆盖相机、选择器、第三方 App 三类往返和连续选择，未再复现；
- 仍建议在现场按「样例 → 拍照 → 相册 → 保存 → AI Benchmark 往返」完整走一遍再开机演示。

## 4. 仍未完成

- [~] Windows 本机运行 Python 桌面版（0.1.1）——2026-11-13 用户决定不再作为验收项，移动端为准。
- [ ] 导师预检查与最终演示形式确认。

## 5. 修订记录

- 2026-11-13 · claude code (local)：一加 9R 覆盖升级 0.1.2，完成根因链与多轮往返复验，归档证据并记录单次计时样本。
