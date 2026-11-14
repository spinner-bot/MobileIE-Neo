# 16 · 桌面版 0.2.1：任意路径可运行与真实 Windows 验收

- 日期：2026-11-14；作者：claude code (local)，Windows 11（本机）+ 服务器 `sqs_A100`
- 对象：**MobileIE-LightLab-Desktop-Python-0.2.1.zip**（2,425,774 B，21 条目）
- 关联：[14 双端 0.2.0 验收](14-three-task-0.2.0-delivery.md)、[15 分支改名](15-branch-rename.md)
- 结论：**桌面版原先在中文路径下无法启动模型；已改为按字节加载，并在真实 Windows + 中文含空格路径下完成 8 项测试与三任务实跑。Android 0.2.0 不变，无需重发。**

## 1. 问题

0.2.0 桌面版把模型路径字符串直接交给 LiteRT：

```python
Interpreter(model_path=str(source), ...)
```

LiteRT 原生层按窄字符打开文件，非 ASCII 路径（如 `F:\资源夹\科研\...`）一律失败：

```
ValueError: Could not open 'F:\...\model.tflite'. The model allocation is null/empty
```

同一文件的字节内容本身完全正常，SHA256 校验也先一步通过——报错发生在原生层重新打开路径时。用户要求的是**包在任何路径都能跑**，不接受"请放到英文目录"这类绕行。

## 2. 修改

三处，均为工具本身：

| 文件 | 修改 |
| --- | --- |
| `desktop/lightlab_tasks.py` | `Engine` 只读一次字节，校验与 `Interpreter(model_content=data)` 共用同一份；不再向原生层传路径 |
| `scripts/run_official_tflite.py` | 兼容层 `Interpreter()` 检测到非 ASCII `model_path` 时改为读字节传 `model_content`，官方脚本本身不改 |
| `desktop/test_desktop.py` | 证据 JSON 固定以 UTF-8 读取（Windows 默认 GBK 会把中文读成乱码而误报）；`native_windows_tested` 改为按平台如实记录；参考数组目录可用 `LIGHTLAB_REFERENCE_DIR` 指定；截图先置顶并按系统缩放换算物理像素 |

顺带收益：模型只读一次，校验与加载用的是同一份字节。

## 3. 真实 Windows 验收

在本机 Windows 11 上原生运行（非 Linux/Xvfb 模拟），工作目录就是含中文的仓库路径：

```
Ran 8 tests in 25.659s
OK
```

| 门控 | 结果 | 证据 |
| --- | --- | --- |
| 核心 + 真实 Tk GUI 8 项 | 8/8 通过，0 跳过 | [desktop-gui.json](../results/desktop-0.2.1/desktop-gui.json) |
| 三任务数值对独立参考 | 9 例，最大原始误差 ≤1e-4、像素差 ≤1 | [desktop-numerics.json](../results/desktop-0.2.1/desktop-numerics.json) |
| 三任务 × 两种展示截图 | 实际查看 | [低光](../results/desktop-0.2.1/lowlight-dynamic.png)、[水下](../results/desktop-0.2.1/underwater-static.png)、[ISP](../results/desktop-0.2.1/isp-static.png) |
| 模型失败保留 baseline、小窗口滚动 | 通过并查看 | [失败态](../results/desktop-0.2.1/desktop-model-failure.png)、[小窗口](../results/desktop-0.2.1/desktop-small-scrolled.png) |

**发布包解包后复验**：把 ZIP 解到 `F:\dev-tools\tmp\中文 路径\mobileie-0.2.1\`（中文 + 空格），直接运行：

| 项 | 结果 |
| --- | --- |
| 三任务加载与推理 | 低光 351.42 ms、水下 348.53 ms、ISP 79.79 ms（模型初始化计入，单次样本） |
| 真实 Tk 界面启动 | 通过，标题栏显示 v0.2.1，见[中文路径启动截图](../results/desktop-0.2.1/desktop-zh-path-launch.png) |
| 模型摘要 | 三份与 `tasks.json` 一致 |

## 4. 发布

| 项 | 值 |
| --- | --- |
| 文件 | `MobileIE-LightLab-Desktop-Python-0.2.1.zip` |
| 大小 | 2,425,774 B（21 条目） |
| SHA256 | `eb6618f7b684798b56dddc0ba8732e7ca3f19cd217fcd6577271221f4f8f6ae7` |
| 服务器 | `/data/Users/zrh/data/MobileIE-Neo/demo/org/releases/0.2.1/` |
| 清单 | [release.json](../results/desktop-0.2.1/release.json) |
| 打包 | [`package-desktop-release.py`](../scripts/package-desktop-release.py)，固定时间戳、确定性字节 |

- **Android 保持 0.2.0**：命令行路径与 Java AssetManager 无关，未受影响，APK 不重发。
- 0.2.0 的桌面 ZIP 保留原字节，不作替换。

## 5. 启动器

仓库内新增 [`微光实验室v0.2.1启动器.bat`](../desktop/微光实验室v0.2.1启动器.bat)，双击即可运行：

- **Python 查找顺序**：环境变量 `MOBILEIE_PYTHON` → 本机演示环境 `F:\dev-tools\python-envs\MobileIE-Demo\Scripts\python.exe` → PATH 上的 `python` → `py -3`；
- 找不到 Python 或同目录缺 `desktop.py` 时给出中文提示并暂停，不会一闪而过；
- 带参数 `check` 时只做依赖自检，不打开界面：`微光实验室v0.2.1启动器.bat check`；
- 程序异常退出时提示安装依赖的命令。

文件按 **GBK + CRLF** 保存（cmd.exe 需要 CRLF，中文提示按本地代码页显示），因此在 `.gitattributes` 中把 `*.bat` 声明为 `binary`，避免被统一换行规则改写。已验证仓库内字节与工作区一致（1270 B、44 个 CRLF）。

本机桌面快捷方式 `微光实验室v0.2.1.lnk` 指向该 bat，工作目录为 `demo/org/desktop`，图标取演示 Python，窗口最小化启动。实测双击后 9 秒内出现「微光实验室 · 三任务对比」窗口。

启动器目前只在仓库中，0.2.1 的 ZIP 内未包含；如需随包发布，应在下次打包时加入 `FILES` 列表。

## 6. 仍未完成

- [x] 一加 9R 覆盖升级 0.2.0 与三任务真机检查——2026-11-14 22:04 完成，见[17 真机验收](17-real-device-0.2.0-9r.md)。
- [ ] 新任务的真机速度（AI Benchmark）测量（跳转与返回已验证，跑分未做）。
- 本次计时为单次样本、CPU 单线程，不与手机或 GPU 数字混用。

## 7. 修订记录

- 2026-11-14 · claude code (local)：修复中文路径模型加载，完成真实 Windows 原生测试、中文路径解包复验并发布 0.2.1 桌面包；新增双击启动器与桌面快捷方式。
- 2026-11-14 · claude code (local)：一加 9R 已完成 0.2.0 真机验收，第 6 节首项勾除并指向[17](17-real-device-0.2.0-9r.md)。
