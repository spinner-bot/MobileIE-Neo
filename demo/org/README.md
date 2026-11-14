# 手机与桌面演示 · 微光实验室

`demo.org`负责现场演示，演示改动不合回`main`，不改变研究代码或结论。当前两端统一升级到 **Android / Python 0.2.0**，需求来自[6B13-01会议](../../docs/会议记录/6B13-01/新录音.txt)及随后用户确认。

三种任务与两种展示相互独立：

| 任务 | 模型输入→输出 | 固定传统baseline | 可用输入 |
| --- | --- | --- | --- |
| 低光增强 | 600×400 RGB→同尺寸RGB | Gamma0.5 | 系统拍照、相册、3个样例 |
| 水下增强 | 600×400 RGB→同尺寸RGB | 灰度世界白平衡 | 系统拍照、相册、3个样例 |
| RAW→RGB ISP | 224×224×4→448×448 RGB | 双线性去马赛克+白平衡+sRGB | 配套448×448 RGGB `.rggb`、3个真实RAW样例 |

- **动态对比**：原图↔baseline、原图↔MobileIE两个窗口并排，分界线同步。
- **静态总览**：原图、MobileIE、baseline独立三图。
- 展示上方、导出组合图与JSON明确标注**本次处理任务**。切换任务清除旧结果。
- 每套模型只有一个图像输入、没有原生可调控制量，按用户确认**不做参数面板**。

RGB输入使用方向校正后的600×400共同取景范围。ISP不能用普通照片或任意DNG替代RAW；其原图面板是明确标记的RAW强度可视化。传统方法固定，未增加曝光/Gamma滑块或额外调节步骤。

Android保留拍照、系统选图、保存、进程恢复与AI Benchmark跳转；Python桌面处理已有文件。AI Benchmark独立测速，微App展示效果，二者不自动交换图片或跑分。原低光模型字节不变；新增水下/ISP由官方冻结权重转换，均已做官方参考数值核对。

本版服务器验收范围：Android15模拟器26项仪器测试、低光6组及三任务9组数值；Linux/Xvfb桌面8项测试、9个任务样例与界面检查。**0.2.0一加9R和真实Windows仍需本地彩排**；0.1.2一加9R已通过的记录保留，不能直接替代本版实测。

入口与交付：

| 内容 | 位置 |
| --- | --- |
| 本版范围、取消参数面板与阶段门控 | [13设计确认](docs/13-three-task-and-display-plan.md) |
| 0.2.0交付、测试证据与安装包索引 | [14验收报告](docs/14-three-task-0.2.0-delivery.md) |
| Android构建与验收命令 | [App说明](app/README.md) |
| Python启动与输入限制 | [桌面快速说明](desktop/QUICKSTART.txt) |
| 手机操作、AI Benchmark配合与彩排清单 | [中文操作说明](demo-package/微光实验室操作说明.txt) |
| 开发环境配置 | [06安卓环境报告](docs/06-android-environment.md) |
| 模型/样例/许可来源 | [任务来源](app/task-asset-sources.json)、[credits](app/app/src/main/assets/credits.txt) |
| 既有一加9R实测 | [0.1.1检查](docs/10-real-device-0.1.1.md)、[0.1.2复验](docs/12-android-0.1.2-reverify.md) |
| 早期方案及交付 | [01方案](docs/01-demo-plan.md)、[09首版交付](docs/09-app-v011-final.md)、[11修复报告](docs/11-android-0.1.2-state-fix.md) |
| 历史AI Benchmark结果 | [设备记录](data/test-devices.md)、[手机证据](results/phone/) |

最终验收为**2026-11-16周一上午**，导师主讲，用户协助操作。11月13日晚导师已检查并提出三任务/显示改进；按会议安排继续准备11月14日晚预检查、11月15日江宁现场准备，具体时刻和会场以通知为准。一加13按既定安排只准备静态图，不安装演示App。

版本管理：文件/模块级小步密集提交，阶段测试验收后推送`demo.org`。同步使用`git fetch`及显式`git merge --no-ff`；禁止fast-forward合并、rebase、squash和强推。大安装包及运行资产保存在zrh自有数据目录，仓库保存索引、SHA256、报告、截图和测试日志。旧版交付保持原字节和摘要。
