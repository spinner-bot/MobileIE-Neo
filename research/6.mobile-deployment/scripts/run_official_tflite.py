#!/usr/bin/env python3
"""薄包装：复用官方 test_TFLite_RGB.py 完成 TFLite 推理。

只负责传路径，不重写预处理/推理/保存逻辑。需要环境中已安装 TensorFlow
（官方脚本依赖 `from tensorflow import lite`）。

用法：
    python run_official_tflite.py \
        --model MobileIE/pretrain/lolv1.tflite \
        --input MobileIE/data/LOLdataset/eval15/low \
        --output research/6.mobile-deployment/results/lol_eval15
"""

import argparse
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OFFICIAL = REPO_ROOT / "MobileIE" / "test_TFLite_RGB.py"


def load_official(path: Path):
    if not path.exists():
        sys.exit(f"找不到官方脚本：{path}")
    spec = importlib.util.spec_from_file_location("official_tflite_rgb", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("tensorflow"):
            sys.exit(
                "当前环境没有 TensorFlow，无法运行官方脚本。\n"
                "请按 docs/03-official-tflite-inference.md 新建演示环境并安装 tensorflow-cpu。"
            )
        raise
    return module


def check_input_size(folder: Path):
    try:
        from PIL import Image
    except ModuleNotFoundError:
        return
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        width, height = Image.open(path).size
        if (width, height) != (600, 400):
            print(
                f"警告：{path.name} 尺寸为 {width}x{height}，模型要求 600x400；"
                "请先用 prepare_demo_images.py resize 处理。"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help=".tflite 路径")
    parser.add_argument("--input", required=True, help="输入图片目录")
    parser.add_argument("--output", required=True, help="输出图片目录")
    parser.add_argument("--official", default=str(DEFAULT_OFFICIAL), help="官方脚本路径")
    args = parser.parse_args()

    model = Path(args.model)
    input_dir = Path(args.input)
    if not model.exists():
        sys.exit(f"找不到模型：{model}")
    if not input_dir.is_dir():
        sys.exit(f"找不到输入目录：{input_dir}")

    check_input_size(input_dir)
    official = load_official(Path(args.official))

    interpreter = official.load_model(str(model))
    official.print_model_structure(interpreter)
    official.inference_and_save_results(interpreter, str(input_dir), args.output)
    print(f"done: {args.output}")


if __name__ == "__main__":
    main()
