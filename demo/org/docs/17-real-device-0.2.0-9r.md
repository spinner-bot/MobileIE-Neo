# 17 · 安卓 0.2.0 一加 9R 真机验收

- 执行：2026-11-14 22:04–22:24 · claude code (local)，Windows 11 + adb（USB，设备 `b62e7c1`）
- 对象：**微光实验室 0.2.0**（`dev.zrh.mobileie`，versionCode 4）
- 关联：[14 双端 0.2.0 交付](14-three-task-0.2.0-delivery.md)、[12 0.1.2 真机复验](12-android-0.1.2-reverify.md)、证据 [`results/phone-0.2.0-9r/`](../results/phone-0.2.0-9r/)
- 结论：**覆盖升级成功、旧数据保留；三任务在真机上均能处理；相机、相册、RAW 选择器三类往返与 AI Benchmark 跨应用往返全部通过，返回后状态与结果不丢。已补做本地拍照与相册照片的完整处理链。真机 AI Benchmark 跑分未测。**

## 1. 升级

| 项 | 值 |
| --- | --- |
| 设备 | 一加 9R（`LE2100`），Android 14，1080×2400 |
| 安装 | `adb install -r` 覆盖安装，未卸载、未清数据 |
| 版本 | `versionName=0.2.0`、`versionCode=4` |
| 时间 | `firstInstallTime` 2026-11-13 09:22:42（0.1.x 时期），`lastUpdateTime` 2026-11-14 22:04:40 |
| 旧数据 | `Download/MobileIE/` 与 `Pictures/MobileIE/` 中 0.1.2 记录保留，`app_version` 仍为 0.1.2 |

冷启动后恢复上次会话（低光样例），属预期行为。

## 2. 结果

每一步以 UI 树（`uiautomator dump`）的文本与 `enabled` 属性核对状态，不只看屏幕观感。

| 步骤 | 结果 | 证据 |
| --- | --- | --- |
| 三任务逐一处理（低光 / 水下 / ISP） | 均「处理完成」，任务标题与 baseline 名称正确绑定 | [00](../results/phone-0.2.0-9r/00-low-light-home.png) |
| ISP 任务的「现场拍照」 | `enabled=false`，点击不响应——符合设计（ISP 只收配套 RAW） | [01](../results/phone-0.2.0-9r/01-isp-camera-disabled.png) |
| 往返 1：现场拍照 → 系统相机 | `com.oneplus.camera/ServiceCameraActivity` 正常拉起 | [02](../results/phone-0.2.0-9r/02-camera-open.png) |
| 往返 1 取消：返回键 | 回到本 App，提示「已取消拍照，保留此前结果」，结果未丢 | [03](../results/phone-0.2.0-9r/03-camera-cancelled.png) |
| 往返 1 拍照：快门 → 确认 | 照片回传，来源改为「现场拍摄 · 600×400 共同取景范围」 | [04](../results/phone-0.2.0-9r/04-camera-capture-confirm.png)、[05](../results/phone-0.2.0-9r/05-camera-photo-ready.png) |
| **本机照片完整处理** | 开始对比 → 处理完成，两方法各自出图 | [06](../results/phone-0.2.0-9r/06-camera-photo-processed.png) |
| 往返 2：相册选图 → `DocumentsUI` 选择器 | 正常打开；返回键取消后提示「已取消选图，保留此前结果」 | [07](../results/phone-0.2.0-9r/07-gallery-cancelled.png) |
| 往返 2 选图 | 选中相册照片后来源改为「相册照片 · 600×400 共同取景范围」 | [08](../results/phone-0.2.0-9r/08-gallery-picked.png) |
| **相册照片完整处理** | 处理完成，MobileIE 提亮、Gamma 0.5 偏品红，与低光任务行为一致 | [09](../results/phone-0.2.0-9r/09-gallery-photo-processed.png) |
| 往返 3：ISP「导入 RAW」 | 选择器按任意文件列出（`.rggb` 不是图片 MIME）；取消后保留结果 | [10](../results/phone-0.2.0-9r/10-raw-picker.png)、[11](../results/phone-0.2.0-9r/11-raw-cancelled.png) |
| 往返 3 导入 `Download/MobileIE/*.rggb`（401,436 B） | 来源改为「导入 RAW · 448×448 RAW 强度可视化」 | [12](../results/phone-0.2.0-9r/12-raw-imported.png) |
| **导入 RAW 完整处理** | 处理完成，与内置同源样例输出一致 | [13](../results/phone-0.2.0-9r/13-raw-imported-processed.png) |
| 内置样例 01 → 02 → 03 | 对话框逐项切换，原图与说明同步刷新，旧结果清空 | [14](../results/phone-0.2.0-9r/14-sample-03-selected.png) |
| 水下任务 + 现场拍摄（设备上同时出现的人工操作，非本次自动步骤） | 拍照后处理完成，两条分界线同步 | [15](../results/phone-0.2.0-9r/15-underwater-camera-processed.png) |
| 往返 4：AI Benchmark | 首次跳转由系统确认「微光实验室 wants to open AI Benchmark」；确认后 `org.benchmark.demo/StartActivity` 正常打开 | [16](../results/phone-0.2.0-9r/16-aibench-confirm-dialog.png)、[17](../results/phone-0.2.0-9r/17-aibench-opened.png) |
| 从 AI Benchmark 返回 | 回到本 App，任务、结果与计时（MobileIE 48.6 ms / baseline 90.0 ms）都在 | [18](../results/phone-0.2.0-9r/18-returned-with-timings.png) |
| 保存对比结果 | 提示已存相册与记录；`Pictures/MobileIE/` 生成 comparison/baseline/mobileie/input PNG，`Download/MobileIE/` 生成 `.rggb` 与 record.json | [19](../results/phone-0.2.0-9r/19-saved-to-gallery.png) |

### 2.1 本次计算记录

[`record-isp-sample1.json`](../results/phone-0.2.0-9r/record-isp-sample1.json)：`app_version=0.2.0`、`task_id=isp`、`input_shape=[1,224,224,4]`、`output_shape=[1,448,448,3]`、`native_control_inputs=[]`、`runtime=LiteRT 1.4.2 CPU built-in, 4 threads`、`model_sha256` 与发布一致。分段：preparation 13.31、packing 19.13、inference 48.59、rendering 41.79、总计 204.94 ms；baseline 90.00 ms。

单次样本、CPU 4 线程，**不与桌面或 GPU 数字混用，也不是 AI Benchmark 跑分**。

## 3. 设备与判定口径

- 表中除水下任务一行外，均为 adb 注入点击与按键完成；系统相机/相册/文件选择器都是真实系统组件，未用 stub。
- 会话中途设备上也出现过本机操作者自己的界面状态（水下任务的现场拍摄结果，22:20 前后），该截图作为补充证据保留，不计入自动步骤，用于说明同一台设备上人工操作也能得到一致结果。
- 相机预览在多数时刻被遮挡或对着室内，照片内容本身不含目标场景，因此**本报告只验证链路与状态，不评价画质**。
- 「处理完成」只表示两条通路都跑完并出图，不是质量结论。

## 4. 仍未完成

- [ ] 真机 AI Benchmark 跑分（本次只验证跳转与返回，未触发评测）。
- [ ] 新任务在 AI Benchmark 中的模型速度对照。
- [ ] 弱光真实场景拍照、长时间连续使用与热降频观察。
- [ ] 导师 2026-11-16 现场验收。

## 5. 修订记录

- 2026-11-14 · claude code (local)：0.2.0 覆盖升级；三任务处理、三类选择器往返、本机拍照/相册完整处理、AI Benchmark 跨应用往返、保存与记录全部实测通过。
