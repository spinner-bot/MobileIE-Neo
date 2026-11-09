"""闪烁来源拆解：分别开关泊松噪声 / JPEG 压缩 / 曝光抖动。

对每种退化组合，统计输入与输出的对齐帧间残差，回答"哪类输入扰动贡献最大"。

用法：python research/3.video-flicker/scripts/flicker_ablation.py [--device cuda]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

W, H = 640, 480
PAN = 4
N_FRAMES = 60
OUT = 'research/3.video-flicker/results_ablation'


def frame_at(img, t):
    x = min(PAN * t, img.shape[1] - W)
    y0 = (img.shape[0] - H) // 2
    return img[y0:y0 + H, x:x + W]


def degrade(gt_bgr, rng, noise=False, jpeg=False, exposure=0.0):
    x = gt_bgr.astype(np.float32) / 255.0
    x = np.clip((x ** 2.2) * 0.35, 0, 1)
    if noise:
        x = rng.poisson(x * 40.0) / 40.0
        x = np.clip(x + rng.normal(0, 0.008, x.shape), 0, 1)
    if exposure:
        x = np.clip(x * (1.0 + rng.uniform(-exposure, exposure)), 0, 1)
    img = (x * 255).astype(np.uint8)
    if jpeg:
        ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def to_tensor(bgr, device):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)


def enhance(net, bgr, device):
    with torch.no_grad():
        out = net(to_tensor(bgr, device)).clamp(0, 1)[0]
    arr = (out.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def aligned_diff(prev, cur):
    a = np.roll(cur, PAN, axis=1)[:, :W - PAN].astype(np.float32)
    b = prev[:, :W - PAN].astype(np.float32)
    return float(np.abs(a - b).mean())


def run(net, img, device, rng, **kw):
    outs, ins, drift = [], [], []
    prev_o = prev_i = None
    for t in range(N_FRAMES):
        gt = frame_at(img, t)
        inp = degrade(gt, rng, **kw)
        out = enhance(net, inp, device)
        outs.append(float(out.mean())); ins.append(float(inp.mean()))
        drift.append(float(out.mean()) - float(gt.mean()))
        if prev_o is not None:
            yd = aligned_diff(prev_o, out)
            idf = aligned_diff(prev_i, inp)
            yield yd, idf, float(out.mean()) - float(inp.mean())
        prev_o, prev_i = out, inp


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default=os.path.join(root, 'MobileIE'))
    ap.add_argument('--weights', default='pretrain/lolv1_best_slim.pkl')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    repo = os.path.abspath(args.repo)
    sys.path.insert(0, repo)
    os.chdir(repo)
    from model.lle import MobileIELLENetS

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    net = MobileIELLENetS(channels=12)
    net.load_state_dict(torch.load(args.weights, map_location='cpu', weights_only=True))
    net.eval().to(device)

    scene = os.path.join(root, 'research/3.video-flicker/assets/source_scene.jpg')
    img = cv2.cvtColor(np.array(Image.open(scene).convert('RGB')), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (1600, 1200), interpolation=cv2.INTER_AREA)

    modes = {
        'none':            dict(),
        'noise':           dict(noise=True),
        'jpeg':            dict(jpeg=True),
        'exposure':        dict(exposure=0.03),
        'noise+jpeg':      dict(noise=True, jpeg=True),
        'noise+jpeg+exp':  dict(noise=True, jpeg=True, exposure=0.03),
    }

    results = {}
    for name, kw in modes.items():
        rng = np.random.default_rng(0)
        outs, ins = [], []
        for yd, idf, _ in run(net, img, device, rng, **kw):
            outs.append(yd); ins.append(idf)
        results[name] = {
            'output_flicker': float(np.mean(outs)),
            'input_flicker': float(np.mean(ins)),
            'amplification': float(np.mean(outs) / max(np.mean(ins), 1e-6)),
        }
        print('%-16s input %7.3f | output %7.3f | x%.2f' % (
            name, results[name]['input_flicker'], results[name]['output_flicker'],
            results[name]['amplification']))

    out_dir = os.path.join(root, OUT)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(results, open(os.path.join(out_dir, 'ablation.json'), 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    names = list(modes)
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar(x - w / 2, [results[n]['input_flicker'] for n in names], w, label='输入', color='#8b96a5')
    ax.bar(x + w / 2, [results[n]['output_flicker'] for n in names], w, label='MobileIE 输出', color='#bc4c00')
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel('对齐帧间残差 (0-255)')
    ax.set_title('闪烁来源拆解：不同输入扰动下的时序不一致')
    ax.grid(axis='y', alpha=.3); ax.legend()
    for i, n in enumerate(names):
        ax.annotate('x%.1f' % results[n]['amplification'],
                    (i + w / 2, results[n]['output_flicker']), ha='center', va='bottom', fontsize=9,
                    color='#bc4c00')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'ablation.png')); plt.close(fig)
    print('wrote', os.path.join(out_dir, 'ablation.png'))


if __name__ == '__main__':
    main()
