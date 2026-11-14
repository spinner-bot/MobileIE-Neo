# 14 · Android / Python 0.2.0 双端升级与验收

日期：2026-11-14；作者：Codex  (sqs_A100)。

两端均完成三任务、双展示及任务标注升级，服务器模拟交付验收通过。源码阶段测试通过后已推送`79d0eeec`，本报告随交付阶段提交推送。**本版一加9R与真实Windows彩排尚未执行**，不能把模拟验收当作导师或最终现场验收。

## 本次完成的行为

| 任务 | 官方模型 | 输入→输出 | 固定baseline |
| --- | --- | --- | --- |
| 低光增强 | 原有LOL TFLite，字节未变 | 600×400 RGB→同尺寸 | Gamma0.5 |
| 水下增强 | `uieb_best_slim.pkl`冻结权重转换 | 600×400 RGB→同尺寸 | 灰度世界白平衡 |
| RAW→RGB ISP | `zrr_best_slim.pkl`冻结权重转换 | packed224×224×4→448×448 RGB | 双线性去马赛克、灰度世界白平衡、sRGB编码 |

动态模式并排显示“原图↔baseline”和“原图↔MobileIE”，拖动任一窗口，两条分界线同步。静态模式保留独立三图。处理后，画面上方显示**本次任务**，绑定实际结果记录；切换任务会清除旧结果，准备对应样例。导出的组合图和JSON同样标注任务与方法。

逐任务核查原生接口：LLE/UIE的`MobileIELLENetS.forward(x)`、ISP的`MobileIEISPNetS.forward(x)`均只有图像张量；三份移动模型也只有一个图像输入，没有额外控制输入。按用户纠正取消参数面板，没有曝光、Gamma或混合强度滑块。baseline使用固定常量。

Android保留系统拍照、相册、内置样例、独立保存、AI Benchmark跳转及进程恢复。Python沿用脚本运行方式，无需Windows原生开发工具。两端统一0.2.0，Android为versionCode4，同旧版开发签名，可覆盖升级。

## RAW与来源边界

RGB图片按原有共同取景流程得到600×400输入；ISP不能把普通JPEG/PNG或任意DNG当作RAW。ISP仅接受配套448×448 Huawei P20 RGGB `.rggb`，包含原始uint16像素及固定传感器说明。三个样例来自Zurich RAW-to-RGB作者发布包的`test/huawei_raw/{0,1,10}.png`，无生成、改值或缩放。导入/导出保留原RAW字节。

模型按官方代码打包四通道并除4095，不减黑电平。原图窗口明确标记为**RAW强度可视化**，以10bit强度显示马赛克，不冒充RGB原片。传统ISP是上述固定简单参考，未实现相机专用色彩矩阵、完整商业ISP或自动曝光；不能由样例观感宣称其代表所有传统ISP。

水下样例为UIEB公开分发的前三个validation输入，低光样例沿用旧版。APK及Python ZIP均带许可、学术演示用途说明及[来源/摘要](../app/task-asset-sources.json)。本次属于演示与推理一致性检查，不是新的论文质量评测。

## 测试与验收结果

| 门控 | 实际结果 | 证据 |
| --- | --- | --- |
| UIE/ISP官方PyTorch→TFLite | 14/14通过；原始输出绝对误差≤1e-4，8bit像素差≤1 | [转换报告](../results/task-models-0.2.0/conversion-report.json) |
| Android仪器回归 | **26/26通过**，187.719秒 | [原始日志](../results/app-0.2.0/instrumentation.txt) |
| 原低光独立数值核对 | 6/6通过，保留原模型/脚本参考 | [数值报告](../results/app-0.2.0/numerical-report.json) |
| Android三任务真实样例 | 9/9通过；最大原始误差1.3113e-6，像素差≤1；输入/RAW打包精确一致 | [任务数值报告](../results/app-0.2.0/task-numerical-report.json) |
| Python核心及真实Tk GUI | **8/8通过，0跳过**，12.692秒；9个样例数值通过，最大原始误差1.0431e-6 | [桌面日志与截图](../results/desktop-0.2.0/) |
| 相机前台App进程死亡 | 低光、水下分别通过；返回照片仍能处理且任务不变 | [低光](../results/app-0.2.0/camera-process-recovery-lowlight.json)、[水下](../results/app-0.2.0/camera-process-recovery-underwater.json) |
| RAW冷进程恢复 | 任务、结果及记录保留，RGB相机禁用 | [恢复报告](../results/app-0.2.0/raw-cold-restore.json) |
| 实际APK覆盖升级 | 从0.1.2生成旧结果后安装0.2.0；照片/旧图/记录保留，重处理写入新版任务与版本 | [升级报告](../results/app-0.2.0/upgrade-from-0.1.2.json) |
| 界面实际检查 | 两端各三任务×两种展示；标题、并置、同步、静态、失败提示及桌面小窗口通过 | [视觉检查](../results/app-0.2.0/visual-review.json) |
| 视频解码 | 额外1项通过；63.392秒、720×1560、真实中间帧解码 | [解码报告](../results/app-0.2.0/video-decoder.json) |
| 构建/Lint/签名/对齐 | 构建通过，0错误；签名兼容旧版，16KB zipalign通过 | [Lint](../results/app-0.2.0/lint-results-debug.txt)、[签名](../results/app-0.2.0/signing-compatibility.txt) |
| Python ZIP独立解压 | 新模块和全部资产齐全，9次推理/保存通过，RAW导出字节一致 | [交付包检查](../results/app-0.2.0/package-verification.json) |
| 发布完整性 | APK/ZIP/视频SHA256、ZIP CRC、嵌入APK及三模型通过；旧三版文件未改变 | [发布清单](../results/app-0.2.0/release.json) |

仪器测试覆盖三任务切换、两组滑动同步、两种展示、真实系统RAW选择器、RGB拒绝路径、导出回读、失败保留baseline、取消相机/选图、后台完成及反复回退/重建。仍有9条非阻断Lint警告：6条固定依赖有新版提示、3条中文演示文本国际化提示，未在验收前更换运行时。

转换初期的失败尝试保留在`results/task-models-0.2.0/`，没有降低误差门槛或调整权重。最终使用数学等价的分层平均减少浮点累积误差，ISP PixelShuffle保持原通道顺序。ISP合成资格样例使用所支持传感器的黑电平64至白值1023再除4095，并另含三个真实RAW；早期0..1满幅浮点压力输入超出该传感器标度，不能据本次通过推广为任意RAW范围/相机的保证。

AI Benchmark的未安装路径及外部包启动路径通过；后者使用明确标记的临时stub，测试后已移除，**不代表实际AI Benchmark界面或新增模型测速通过**。原有一加9R低光GPU约5.8ms属于历史结果；新任务的真机速度需要单独测量。

## 下载与启动

交付目录：`/data/Users/zrh/data/MobileIE-Neo/demo/org/releases/0.2.0/`。大文件依项目约定存放个人资产目录，Git保存源码、证据和摘要索引。

| 文件 | 字节数 | 用途 |
| --- | ---: | --- |
| [MobileIE-LightLab-0.2.0.apk](/data/Users/zrh/data/MobileIE-Neo/demo/org/releases/0.2.0/MobileIE-LightLab-0.2.0.apk) | 19,624,028 | 安卓安装，约19.6MB / 18.7MiB |
| [Android-Demo-0.2.0.zip](/data/Users/zrh/data/MobileIE-Neo/demo/org/releases/0.2.0/MobileIE-LightLab-Android-Demo-0.2.0.zip) | 13,174,946 | APK、录像、操作说明、三模型及RAW样例 |
| [Desktop-Python-0.2.0.zip](/data/Users/zrh/data/MobileIE-Neo/demo/org/releases/0.2.0/MobileIE-LightLab-Desktop-Python-0.2.0.zip) | 2,425,651 | Python双模块、依赖清单、启动脚本及全部资产 |
| [emulator-demo.mp4](/data/Users/zrh/data/MobileIE-Neo/demo/org/releases/0.2.0/MobileIE-LightLab-0.2.0-emulator-demo.mp4) | 4,015,487 | 三任务两展示的模拟操作录像 |

APK SHA256：`d28471059033fb4e77367a4f73d97a7bb9284bca7354662fa93eecaf74f79b4b`。其余摘要见[release.json](../results/app-0.2.0/release.json)。

安卓：`adb install -r MobileIE-LightLab-0.2.0.apk`。桌面：解压并保持文件相对位置，使用既有演示Python执行`python desktop.py`，或配置`MOBILEIE_PYTHON`后运行`run.cmd`；新模块`lightlab_tasks.py`须随包保留。详见[桌面说明](../desktop/QUICKSTART.txt)及[手机说明](../demo-package/微光实验室操作说明.txt)。

## 结束条件与现场接续

本阶段M0–M4已完成：需求和模型契约冻结、三任务模型/来源核对、双端实现、模拟与数值验收、交付包验证及报告。小步提交，源码和交付阶段分别在测试通过后推送；同步只用fetch及`--no-ff`，不rebase/squash/强推。

本次完成的是软件升级及服务器交付验收，后续现场事项仍按证据闭环：

1. 一加9R覆盖升级0.2.0，三任务各处理一张，确认两种展示、同步滑动、任务标题及保存；低光拍照/相册与AI Benchmark往返后反复切样例。0.1.2已实测通过的[历史记录](12-android-0.1.2-reverify.md)保留。
2. 真实Windows使用既有Python，核对三任务、原生对话框、中文字体、DPI及导出；Linux/Xvfb通过不能替代此项。
3. 按6B13-01准备11月14日晚导师再检查、11月15日江宁现场设备/投屏/串场，11月16日周一上午最终验收。导师主讲，用户操作，具体时间与会场仍以通知为准。一加13不安装App，静态图备用。
