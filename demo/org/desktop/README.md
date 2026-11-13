# 微光实验室 · Python 桌面版

用于 Windows 电脑上的图片效果演示：选图或内置样例、MobileIE / 固定 Gamma 0.5 / 原图、同位置滑动对比、三图总览、PNG 与 JSON 保存。现场拍照由 Android 版完成，桌面首版处理已有照片。

## 在 Windows 直接运行

解压桌面 Python 演示包，保留 `desktop.py`、`requirements.txt` 和 `assets/` 的相对位置。使用已有的演示 Python 环境，安装运行依赖后启动：

```bat
python -m pip install -r requirements.txt
python desktop.py
```

也可设置 `MOBILEIE_PYTHON` 为已有演示环境的 Python 路径后双击 `run.cmd`：

```bat
set MOBILEIE_PYTHON=F:\dev-tools\python-envs\MobileIE-Demo\Scripts\python.exe
run.cmd
```

需要 Python 3.10+，包含 Tcl/Tk（`python -m tkinter` 可检查），以及 NumPy、Pillow、ai-edge-litert。使用 Tkinter 标准界面，没有 Visual Studio、Android SDK 或 C++ 编译要求，不需要转换模型。依赖版本固定在 requirements 中，LiteRT 关闭默认 delegate，沿用此前官方脚本在 Windows 上跑通的 CPU 路径。

## 操作

1. 选择照片或内置暗光样例，检查 600×400 居中处理范围。
2. 点击“开始对比”。左侧原图与右侧所选方法共用同一输入，拖动分界线查看同一位置。
3. 切换 Gamma 0.5 与三图总览。Gamma 是固定幂变换，不包含降噪，不预设哪个方法每张图都更好。
4. “保存结果”会在所选目录下新建 `MobileIE-<编号>` 文件夹，包含成功生成的方法 PNG、带标签三联图和计算记录；失败会回滚临时目录。
5. 查看计算记录可核对模型摘要、实际后端与计时。桌面版使用 CPU **1 线程**以对齐官方默认参考，Android 版为 CPU 4 线程，两者的耗时不作设备速度比较。

窗口支持缩放，小窗口可滚动左侧说明；模型失败时标记未生成并禁用对应选项，Gamma 仍可查看。输入超过 32 MB 或解码失败时提示换图，不将旧输出冒充新结果。

## 已验证与范围

以下结果均来自本次 Linux 测试与虚拟显示模拟：

- 三张既有仓库样例与未修改官方脚本的原始输出差为 0，PNG 逐像素相同；模型摘要与 Android 版一致。
- Gamma 全 256 个输入值、共同输入、透明像素、坏图/大图、PNG/JSON 保存与失败回滚均通过。
- 中文界面在 Linux/Xvfb 的 Tk 8.6 虚拟显示中实际运行，并查看初始、完成、总览、失败和小窗口截图；异步处理、取消、换图清计时、滑动、不可用状态和退出清理通过测试。
- **尚未在真实 Windows 机器运行（包括数值与界面）**；Windows 原生文件对话框、DPI、字体与 DLL 加载需本地最后检查。没有将 Linux 模拟验收写成 Windows 实测。

从完整仓库运行开发核验：

```bash
bash demo/org/scripts/run-desktop-validation.sh
```

该入口使用服务器个人目录中的 Linux 测试依赖与独立虚拟显示，不安装 Windows 开发环境。当前 0.1.1 交付位置、摘要、证据和仍待真机检查的项目见[最终交付报告](../docs/09-app-v011-final.md)。
