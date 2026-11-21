# results 说明

本目录存放 6 号工作的效果验证与演示素材。所有增强结果均由官方
[`MobileIE/test_TFLite_RGB.py`](../../MobileIE/test_TFLite_RGB.py) 生成
（经 [`run_official_tflite.py`](../scripts/run_official_tflite.py) 包装）。

## 目录

| 目录 | 内容 |
| --- | --- |
| `lol_eval15/` | LOLv1 eval15 全部 15 张增强结果（600×400） |
| `compare_lol_eval15/` | 上述 15 张的左右对比图（左输入、右 MobileIE） |
| `demo_input_sample/` | 仓库样例 `sample_01.png` 缩放后的 600×400 输入（位于 `data/`） |
| `demo_output_sample/` | 样例增强结果 |
| `compare_sample/` | 样例左右对比图 |

`data/demo_input_sample/` 为预处理后的输入，单独放在 `data/` 下。

## 验证指标

- 数据集：LOLv1 eval15（15 张）
- PSNR 均值：**23.619 dB**（min 15.650，max 27.618）
- 论文报告值：23.62 dB → **一致**，说明模型与预处理链路正确

计算命令（在项目根目录）：

```bash
F:/dev-tools/python-envs/MobileIE-Demo/Scripts/python.exe - <<'EOF'
import numpy as np
from pathlib import Path
from PIL import Image
out=Path('demo/org/results/lol_eval15')
gt=Path('MobileIE/data/LOLdataset/eval15/high')
ps=[]
for p in sorted(out.glob('*.png')):
    o=np.asarray(Image.open(p).convert('RGB'),dtype=np.float64)/255
    g=np.asarray(Image.open(gt/p.name).convert('RGB'),dtype=np.float64)/255
    ps.append(10*np.log10(1/((o-g)**2).mean()))
print(len(ps), np.mean(ps))
EOF
```

## 复现

```bash
PYD=F:/dev-tools/python-envs/MobileIE-Demo/Scripts/python.exe
PYN=F:/dev-tools/python-envs/MobileIE-Neo/Scripts/python.exe

# 增强
"$PYD" demo/org/scripts/run_official_tflite.py \
  --model MobileIE/pretrain/lolv1.tflite \
  --input MobileIE/data/LOLdataset/eval15/low \
  --output demo/org/results/lol_eval15

# 对比图
"$PYN" demo/org/scripts/prepare_demo_images.py compare \
  --input MobileIE/data/LOLdataset/eval15/low \
  --enhanced demo/org/results/lol_eval15 \
  --output demo/org/results/compare_lol_eval15 \
  --label-enhanced MobileIE --format jpeg
```

对比图默认 JPEG（`--quality 90`）；需要无损时加 `--format png`。

## 备注

- 本机 `tensorflow-cpu` 2.20/2.21 原生库初始化失败，实际使用 `ai-edge-litert` 兼容层，
  并关闭默认 XNNPACK delegate；详见 [`docs/03`](../docs/03-official-tflite-inference.md)。
- 手机实拍素材到位后，放入 `data/demo_input/` 并重复上述流程，输出到
  `results/demo_output/` 与 `results/compare_demo/`。
