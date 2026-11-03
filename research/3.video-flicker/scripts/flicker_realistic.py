"""闪烁实验（真实退化 + 机制消融）。

在已知平移的合成夜景视频上，比较三种条件：
  1. clean      : 仅 gamma 低光（之前的基线）
  2. realistic  : gamma + 泊松暗部噪声 + 读出噪声 + 逐帧 JPEG 压缩（模拟手机夜景视频）
  3. frozen_att : realistic 基础上，冻结 HDPA 全局注意力的输出（检验闪烁是否来自全局池化）

输出：results_realistic/ 下的指标 JSON、逐帧曲线图、帧序列（供 make_videos2.py 合成视频）。

用法：python research/3.video-flicker/scripts/flicker_realistic.py [--device cuda]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

W, H = 640, 480
PAN = 4                 # 每帧平移像素
N_FRAMES = 60
FPS = 30
FRAME_DIR = 'F:/pip-tmp/mobileie_frames_real'
OUT_NAME = 'results_realistic'


def make_image(root):
    p = os.path.join(root, 'research/3.video-flicker/assets/source_scene.jpg')
    return cv2.cvtColor(np.array(Image.open(p).convert('RGB')), cv2.COLOR_RGB2BGR)


def degrade(gt_bgr, rng, realistic=True):
    x = gt_bgr.astype(np.float32) / 255.0
    x = np.clip((x ** 2.2) * 0.35, 0, 1)          # 低光
    if realistic:
        n_photon = 40.0                            # 暗部光子数少 → 泊松噪声明显
        x = rng.poisson(x * n_photon) / n_photon
        x = x + rng.normal(0, 0.008, x.shape)      # 读出噪声
        x = np.clip(x, 0, 1)
    img = (x * 255).astype(np.uint8)
    if realistic:                                  # 逐帧 JPEG 压缩
        ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def frame_at(img, t):
    x = min(PAN * t, img.shape[1] - W)
    y0 = (img.shape[0] - H) // 2          # 上下居中裁剪，得到 640x480 横图
    return img[y0:y0 + H, x:x + W]


def to_tensor(bgr, device):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)


def enhance(net, bgr, device):
    with torch.no_grad():
        out = net(to_tensor(bgr, device)).clamp(0, 1)[0]
    arr = (out.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


class FreezeFirst(nn.Module):
    """首次调用后缓存输出：把全局注意力冻结为第一帧的值。"""
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.cache = None

    def forward(self, x):
        if self.cache is None:
            self.cache = self.inner(x)
        return self.cache


def aligned(a, b):
    return np.roll(b, PAN, axis=1)[:, :W - PAN]


def metrics(prev, cur):
    """返回 (总对齐残差, 局部分量, 全局亮度差)。"""
    ac, ap = aligned(prev, cur), prev[:, :W - PAN]
    d = ac.astype(np.float32) - ap.astype(np.float32)
    g = float(d.mean())
    local = float(np.abs(d - g).mean())
    return float(np.abs(d).mean()), local, g


def run(net, img, device, realistic, rng, save_frames=False):
    total, local, means, glist = [], [], [], []
    in_total, in_local = [], []
    drift = []
    prev = prev_in = None
    for t in range(N_FRAMES):
        gt = frame_at(img, t)
        inp = degrade(gt, rng, realistic)
        out = enhance(net, inp, device)
        means.append(float(out.mean()))
        drift.append(float(out.mean()) - float(gt.mean()))
        if save_frames:
            for kind, fr in (('input', inp), ('output', out), ('ground_truth', gt)):
                d = os.path.join(FRAME_DIR, kind)
                os.makedirs(d, exist_ok=True)
                ok, buf = cv2.imencode('.jpg', fr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                buf.tofile(os.path.join(d, '%04d.jpg' % t))
            if prev is not None:
                diff = np.abs(aligned(prev, out) - out[:, :W - PAN]).astype(np.float32)
                vis = np.clip(diff * 20, 0, 255).astype(np.uint8)
                d = os.path.join(FRAME_DIR, 'diff')
                os.makedirs(d, exist_ok=True)
                ok, buf = cv2.imencode('.jpg', vis)
                buf.tofile(os.path.join(d, '%04d.jpg' % (t - 1)))
        if prev is not None:
            m, l, g = metrics(prev, out)
            total.append(m)
            local.append(l)
            glist.append(g)
        if prev_in is not None:
            mi, li, _ = metrics(prev_in, inp)
            in_total.append(mi)
            in_local.append(li)
        prev, prev_in = out, inp
        if (t + 1) % 20 == 0:
            print('  frame %d/%d' % (t + 1, N_FRAMES))
    return {
        'output_aligned_diff': float(np.mean(total)),
        'output_local_diff': float(np.mean(local)),
        'output_global_shift': float(np.mean(np.abs(glist))),
        'input_aligned_diff': float(np.mean(in_total)),
        'input_local_diff': float(np.mean(in_local)),
        'amplification': float(np.mean(total) / max(np.mean(in_total), 1e-6)),
        'brightness_drift_std': float(np.std(drift)),    # 输出亮度相对 GT 的漂移波动
        'per_frame_aligned': total,
        'per_frame_local': local,
        'per_frame_input_aligned': in_total,
    }


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default=os.path.join(root, 'research/2.reproduction/MobileIE'))
    ap.add_argument('--weights', default='pretrain/lolv1_best_slim.pkl')
    ap.add_argument('--out', default=os.path.join(root, 'research/3.video-flicker', OUT_NAME))
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    repo = os.path.abspath(args.repo)
    sys.path.insert(0, repo)
    os.chdir(repo)
    from model.lle import MobileIELLENetS

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    net = MobileIELLENetS(channels=12)
    net.load_state_dict(torch.load(args.weights, map_location='cpu', weights_only=True))
    net.eval().to(device)

    img = make_image(root)
    img = cv2.resize(img, (1600, 1200), interpolation=cv2.INTER_AREA)
    rng = np.random.default_rng(0)

    os.makedirs(args.out, exist_ok=True)
    report = {}

    print('[1/3] clean')
    report['clean'] = run(net, img, device, realistic=False, rng=rng)
    print('[2/3] realistic')
    report['realistic'] = run(net, img, device, realistic=True, rng=rng, save_frames=True)

    print('[3/3] realistic + frozen global attention')
    net.att = FreezeFirst(net.att)              # 消融：全局注意力冻结
    report['frozen_att'] = run(net, img, device, realistic=True, rng=rng)
    net.att = net.att.inner

    with open(os.path.join(args.out, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out, 'video_jobs.json'), 'w', encoding='utf-8') as f:
        json.dump({'frame_dir': FRAME_DIR, 'out_dir': args.out.replace('\\', '/'), 'fps': FPS,
                   'files': {'input': 'input_lowlight.mp4', 'output': 'output_mobileie.mp4',
                             'ground_truth': 'ground_truth.mp4', 'diff': 'flicker_diff_x20.mp4'},
                   'compare': True}, f, indent=2)

    # 曲线图
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    t = np.arange(1, N_FRAMES)
    plt.figure(figsize=(11, 4.2))
    plt.plot(t, report['realistic']['per_frame_input_aligned'], label='input (noise + JPEG)', lw=1.2)
    plt.plot(t, report['realistic']['per_frame_aligned'], label='MobileIE output', lw=1.4)
    plt.plot(t, report['frozen_att']['per_frame_aligned'], label='output (frozen global attention)', lw=1.2, ls='--')
    plt.plot(t, report['clean']['per_frame_aligned'], label='output (clean input)', lw=1.0, alpha=.7)
    plt.xlabel('frame index'); plt.ylabel('aligned mean abs diff (0-255)')
    plt.title('Temporal inconsistency: input vs MobileIE output')
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'flicker_curve.png'), dpi=130); plt.close()

    print(json.dumps({k: {kk: vv for kk, vv in v.items() if not kk.startswith('per_frame')}
                      for k, v in report.items()}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
