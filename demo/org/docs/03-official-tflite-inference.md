# 03 · 官方 TFLite 脚本使用说明（效果对比图）

- 创建：2026-11-08 · claude code (local)
- 触发：B 方案需要在电脑上复用官方 TFLite 推理脚本，生成低光增强前后对比图。
- 关联：[演示方案](01-demo-plan.md)、[AI Benchmark 手册](02-ai-benchmark-guide.md)
- 上游脚本：[`MobileIE/test_TFLite_RGB.py`](../../MobileIE/test_TFLite_RGB.py)

## 1. 官方脚本做了什么

`MobileIE/test_TFLite_RGB.py` 是仓库自带的 TFLite 推理脚本，流程与端侧一致：

1. `tf.lite.Interpreter` 加载 `.tflite`；
2. 读取图片 → RGB → `float32 / 255.0` → 增加 batch 维（NHWC）；
3. `set_tensor` → `invoke` → `get_tensor`；
4. 裁剪到 [0, 1] → ×255 → 保存为 PNG。

注意：脚本**不做 resize**，图片尺寸必须与模型输入一致（400×600，即 PIL 尺寸 600×400）。

## 2. 脚本内的硬编码路径

```python
model_path          = './LLE.tflite'
input_image_folder  = './lowlight/LOLdataset/eval15/low'
output_image_folder = './experiments/results'
```

因此需要在 `MobileIE/` 目录下运行，并满足上述路径，或按 §4 用包装脚本传入实际路径。

## 3. 运行环境

当前项目虚拟环境 `F:\dev-tools\python-envs\MobileIE-Neo` **没有 TensorFlow / tflite_runtime**，且已安装 torch/numpy，直接装 TensorFlow 可能改动 numpy 版本、影响 3/4 号工作。

**建议新建独立演示环境**（不影响研究环境）：

```bash
# 以 Python 3.13 为例；若 TensorFlow 暂无对应轮子，改用 3.11/3.12
py -3.13 -m venv F:/dev-tools/python-envs/MobileIE-Demo

# 把 pip 缓存/临时目录放到 F 盘（C 盘空间紧张时）
export PIP_CACHE_DIR=F:/pip-cache
export TMP=F:/pip-tmp
export TEMP=F:/pip-tmp

F:/dev-tools/python-envs/MobileIE-Demo/Scripts/python.exe -m pip install --upgrade pip
F:/dev-tools/python-envs/MobileIE-Demo/Scripts/python.exe -m pip install tensorflow-cpu pillow numpy
```

验证：

```bash
F:/dev-tools/python-envs/MobileIE-Demo/Scripts/python.exe -c "import tensorflow as tf; print(tf.__version__)"
```

> 若 `tensorflow-cpu` 在 Python 3.13 无可用轮子，优先换 Python 3.12 建环境；不建议为了演示去动研究环境。

### 3.1 本机实测结论（2026-11-08）

| 方案 | 结果 |
| --- | --- |
| `tensorflow-cpu==2.21.0` | 失败：`_pywrap_tensorflow_common.dll INITIALIZATION FAILED (0x45A)` |
| `tensorflow-cpu==2.20.0` | 失败：`_pywrap_tensorflow_internal` 动态链接库初始化失败 |
| `ai-edge-litert==2.3.0` | 可用，但默认 XNNPACK delegate 在 `allocate_tensors()` 时段错误 |

最终采用：**ai-edge-litert + `OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES`**。`run_official_tflite.py` 会自动完成兼容层注入，无需改官方脚本：

```python
from ai_edge_litert.interpreter import Interpreter, OpResolverType
Interpreter(model_path=..., experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
```

PC 出图不追求速度，关闭默认 delegate 不影响结果正确性。

## 4. 推荐运行方式

### 方式 A：不改官方脚本，准备约定路径

```bash
cd MobileIE
cp pretrain/lolv1.tflite LLE.tflite
mkdir -p lowlight/LOLdataset/eval15/low
cp data/LOLdataset/eval15/low/*.png lowlight/LOLdataset/eval15/low/
F:/dev-tools/python-envs/MobileIE-Demo/Scripts/python.exe test_TFLite_RGB.py
```

输出在 `MobileIE/experiments/results/`。

### 方式 B：包装脚本复用官方函数（推荐）

`scripts/run_official_tflite.py` 通过 `import` 调用官方脚本里的
`load_model` / `print_model_structure` / `inference_and_save_results`，
只负责传路径，不重写推理逻辑：

```bash
F:/dev-tools/python-envs/MobileIE-Demo/Scripts/python.exe \
  demo/org/scripts/run_official_tflite.py \
  --model MobileIE/pretrain/lolv1.tflite \
  --input MobileIE/data/LOLdataset/eval15/low \
  --output demo/org/results/lol_eval15
```

> 包装脚本属于“传参胶水”，核心推理仍是官方实现；如不需要可直接用方式 A。

**已验证结果（2026-11-08）**：eval15 共 15 张，输出 PSNR 均值 **23.619 dB**，与论文报告的 23.62 dB 一致。

## 5. 手机实拍照片

手机原图通常远大于 400×600，必须先预处理：

```bash
# 生成 400×600（宽 600、高 400）的输入图
F:/dev-tools/python-envs/MobileIE-Neo/Scripts/python.exe \
  demo/org/scripts/prepare_demo_images.py \
  --input F:/path/to/phone/photos \
  --output demo/org/data/demo_input
```

然后用方式 B 对 `data/demo_input` 推理，输出到 `results/demo_output`。

## 6. 效果预期

- `lolv1.tflite` 在 LOLv1 上训练，擅长**低光提亮**；
- 对正常曝光的照片，输出可能过亮或偏色，属正常；
- 现场素材应选择明显偏暗的照片，对比最直观；
- 水下增强（UIEB）目前只有 `uieb_best_slim.pkl`，需要重新走 ONNX → TF → TFLite 转换，不在本次 B 方案范围内。

## 7. 备选：ONNX Runtime（仅当 TensorFlow 装不上）

仓库有同一权重的 `MobileIE/pretrain/lolv1.onnx`，可用 `onnxruntime` 跑同一模型出图。区别：

- 优点：安装小、无 TensorFlow 依赖；
- 缺点：不是官方 TFLite 脚本路径，汇报时需说明“同一权重的 ONNX 版本”。

除非 TensorFlow 环境受阻，否则优先官方 TFLite 路径。

## 8. 输出与命名

建议目录：

```
demo/org/results/
├── lol_eval15/          # 官方 LOL 验证集结果
├── demo_input/          # 手机照片缩放后的 400×600 输入
├── demo_output/         # 手机照片增强结果
└── compare/             # 左右拼接的对比图 / 视频
```

对比图由 `prepare_demo_images.py --compare` 生成（左：原图，右：增强），便于现场直接展示。

## 9. 修订记录

- 2026-11-08 · claude code (local)：创建，整理环境、路径、运行方式与备选方案。
- 2026-11-08 · claude code (local)：补充本机 TensorFlow 初始化失败与 ai-edge-litert 兼容层实测结论；记录 eval15 PSNR 23.619 dB。
