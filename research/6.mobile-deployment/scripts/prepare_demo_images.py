#!/usr/bin/env python3
"""演示图片预处理：缩放到模型输入尺寸，并生成前后对比图。

MobileIE 的 lolv1.tflite 输入固定为 1x400x600x3（高 x 宽），
而手机照片通常更大。本脚本只做“缩放 + 居中裁剪”，不改变推理逻辑。

用法：
    # 1) 把手机照片处理成 600x400 的模型输入
    python prepare_demo_images.py resize --input photos/ --output demo_input/

    # 2) 把原图与增强结果拼成左右对比图
    python prepare_demo_images.py compare \
        --input demo_input/ --enhanced demo_output/ --output compare/
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_size(text: str):
    try:
        w, h = text.lower().split("x")
        return int(w), int(h)
    except ValueError:
        raise argparse.ArgumentTypeError("size 格式应为 宽x高，例如 600x400")


def iter_images(folder: Path):
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def resize_cover(img: Image.Image, width: int, height: int) -> Image.Image:
    """保持比例缩放后居中裁剪到目标尺寸。"""
    img = ImageOps.exif_transpose(img).convert("RGB")
    scale = max(width / img.width, height / img.height)
    new_size = (max(width, round(img.width * scale)), max(height, round(img.height * scale)))
    img = img.resize(new_size, Image.BICUBIC)
    left = (img.width - width) // 2
    top = (img.height - height) // 2
    return img.crop((left, top, left + width, top + height))


def cmd_resize(args):
    src, dst = Path(args.input), Path(args.output)
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in iter_images(src):
        out = resize_cover(Image.open(path), args.width, args.height)
        out.save(dst / (path.stem + ".png"))
        print(f"resize: {path.name} -> {path.stem}.png")
        count += 1
    print(f"done: {count} image(s) -> {dst}")


def load_font(size: int, font_path: str | None):
    if font_path:
        return ImageFont.truetype(font_path, size)
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def cmd_compare(args):
    src, enh, dst = Path(args.input), Path(args.enhanced), Path(args.output)
    dst.mkdir(parents=True, exist_ok=True)
    font = load_font(args.font_size, args.font)
    bar = args.font_size + 16
    gap = 12
    count = 0
    for path in iter_images(src):
        other = enh / (path.stem + ".png")
        if not other.exists():
            print(f"skip: {path.name} (no {other.name} in enhanced folder)")
            continue
        left = resize_cover(Image.open(path), args.width, args.height)
        right = resize_cover(Image.open(other), args.width, args.height)
        canvas = Image.new(
            "RGB",
            (args.width * 2 + gap, args.height + bar),
            (24, 24, 24),
        )
        canvas.paste(left, (0, bar))
        canvas.paste(right, (args.width + gap, bar))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 6), args.label_input, font=font, fill=(255, 255, 255))
        draw.text((args.width + gap + 8, 6), args.label_enhanced, font=font, fill=(255, 255, 255))
        if args.format == "jpeg":
            out = dst / (path.stem + "_compare.jpg")
            canvas.save(out, quality=args.quality, optimize=True)
        else:
            out = dst / (path.stem + "_compare.png")
            canvas.save(out)
        print(f"compare: {path.name} -> {out.name}")
        count += 1
    print(f"done: {count} comparison(s) -> {dst}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    resize = sub.add_parser("resize", help="缩放到模型输入尺寸")
    resize.add_argument("--input", required=True)
    resize.add_argument("--output", required=True)
    resize.add_argument("--width", type=int, default=600)
    resize.add_argument("--height", type=int, default=400)
    resize.set_defaults(func=cmd_resize)

    compare = sub.add_parser("compare", help="生成左右对比图")
    compare.add_argument("--input", required=True, help="原图/输入目录")
    compare.add_argument("--enhanced", required=True, help="增强结果目录")
    compare.add_argument("--output", required=True)
    compare.add_argument("--width", type=int, default=600)
    compare.add_argument("--height", type=int, default=400)
    compare.add_argument("--label-input", default="Input")
    compare.add_argument("--label-enhanced", default="Enhanced")
    compare.add_argument("--font", default=None, help="可选 TTF 路径（显示中文时使用）")
    compare.add_argument("--font-size", type=int, default=20)
    compare.add_argument("--format", choices=["png", "jpeg"], default="png", help="输出格式，jpeg 更省空间")
    compare.add_argument("--quality", type=int, default=90, help="JPEG 质量（1-100）")
    compare.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
