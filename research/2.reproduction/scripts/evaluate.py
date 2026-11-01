"""MobileIE 评测脚本：PSNR + SSIM（不依赖 skimage），可选保存输出图。

用法（在 research/2.reproduction/MobileIE 目录下运行）：
    python ../scripts/evaluate.py -task lle -input data/LOLdataset/eval15/low \
        -gt data/LOLdataset/eval15/high -pretrain pretrain/lolv1_best_slim.pkl [--save out_dir]
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.getcwd())  # 使 model/data 可导入（需在 MobileIE 目录运行）
from model.lle import MobileIELLENetS, MobileIELLENet
from model.isp import MobileIEISPNetS, MobileIEISPNet


def load_image(path, device, isp=False):
    img = Image.open(path)
    arr = np.array(img.convert('RGB') if not isp else img)
    if isp:  # Bayer 原始图按 4 通道 RGGB 打包
        h, w = arr.shape
        arr = arr.reshape(h // 2, 2, w // 2, 2).transpose([1, 3, 0, 2]).reshape([4, h // 2, w // 2])
        arr = arr.astype(np.float32) / 4095
    else:
        arr = arr.transpose([2, 0, 1]).astype(np.float32) / 255
    return torch.from_numpy(arr).unsqueeze(0).to(device)


def psnr(pred, gt):
    mse = ((pred - gt) ** 2).mean()
    return float(10 * torch.log10(1.0 / mse)) if mse > 0 else 100.0


def ssim(pred, gt, sigma=1.5, win=11):
    """高斯窗 SSIM（逐通道均值）"""
    def gaussian_blur(img):
        x = torch.arange(win, dtype=torch.float32, device=img.device) - (win - 1) / 2
        k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        k = (k / k.sum())
        k_h = k.reshape(1, 1, win, 1).expand(img.shape[1], 1, win, 1)
        k_v = k.reshape(1, 1, 1, win).expand(img.shape[1], 1, 1, win)
        out = torch.nn.functional.conv2d(img, k_h, padding=(win // 2, 0), groups=img.shape[1])
        out = torch.nn.functional.conv2d(out, k_v, padding=(0, win // 2), groups=img.shape[1])
        return out

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_x, mu_y = gaussian_blur(pred), gaussian_blur(gt)
    sigma_x = gaussian_blur(pred * pred) - mu_x ** 2
    sigma_y = gaussian_blur(gt * gt) - mu_y ** 2
    sigma_xy = gaussian_blur(pred * gt) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
    return float(ssim_map.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-task', default='lle', choices=['lle', 'isp'])
    ap.add_argument('-input', required=True)
    ap.add_argument('-gt', required=True)
    ap.add_argument('-pretrain', required=True)
    ap.add_argument('-type', default='re_parameterized', choices=['re_parameterized', 'original'])
    ap.add_argument('-save', default=None)
    ap.add_argument('-device', default='cuda')
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    cls = {'lle': (MobileIELLENetS if args.type == 're_parameterized' else MobileIELLENet),
           'isp': (MobileIEISPNetS if args.type == 're_parameterized' else MobileIEISPNet)}[args.task]
    ckpt = torch.load(args.pretrain, map_location='cpu', weights_only=True)
    net = cls(channels=12)
    missing, unexpected = net.load_state_dict(ckpt, strict=False)
    print(f'load: missing={len(missing)} unexpected={len(unexpected)}')
    net.eval().to(device)

    if args.save:
        os.makedirs(args.save, exist_ok=True)
    names = sorted(os.listdir(args.input))
    rows, ps, ss = [], [], []
    for name in names:
        inp = load_image(os.path.join(args.input, name), device, isp=(args.task == 'isp'))
        gt = load_image(os.path.join(args.gt, name), device, isp=(args.task == 'isp'))
        with torch.no_grad():
            out = net(inp).clamp(0, 1)
        p, s = psnr(out, gt), ssim(out, gt)
        ps.append(p)
        ss.append(s)
        rows.append((name, p, s))
        if args.save:
            arr = (out[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)[..., ::-1]
            Image.fromarray(arr).save(os.path.join(args.save, name))
        print(f'{name:24s} PSNR {p:7.3f}  SSIM {s:.4f}')

    print(f'\n>>> n={len(ps)}  mean PSNR {np.mean(ps):.3f}  mean SSIM {np.mean(ss):.4f}')


if __name__ == '__main__':
    main()
