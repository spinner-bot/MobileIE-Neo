"""Training-free 时序稳定化对照：EMA 平滑 与 光流对齐 EMA。

在"已知平移 + 真实退化"的合成夜景视频上，扫描平滑系数 α，
测量闪烁（已知平移对齐残差）与画质（vs GT 的 PSNR / SSIM）的权衡。

用法：python research/3.video-flicker/scripts/temporal_baselines.py [--device cuda]
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
ALPHAS = [0.0, 0.5, 0.7, 0.9]
OUT = 'research/3.video-flicker/results_baselines'


def frame_at(img, t):
    x = min(PAN * t, img.shape[1] - W)
    y0 = (img.shape[0] - H) // 2
    return img[y0:y0 + H, x:x + W]


def degrade(gt_bgr, rng):
    x = gt_bgr.astype(np.float32) / 255.0
    x = np.clip((x ** 2.2) * 0.35, 0, 1)
    x = rng.poisson(x * 40.0) / 40.0
    x = np.clip(x + rng.normal(0, 0.008, x.shape), 0, 1)
    img = (x * 255).astype(np.uint8)
    ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


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


def psnr(a, b):
    mse = ((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean()
    return float(10 * np.log10(255.0 ** 2 / max(mse, 1e-8)))


def ssim(a, b, sigma=1.5, win=11):
    a = a.astype(np.float32) / 255.0
    b = b.astype(np.float32) / 255.0
    x = np.arange(win) - (win - 1) / 2
    k = np.exp(-(x ** 2) / (2 * sigma ** 2)); k /= k.sum()
    blur = lambda im: cv2.sepFilter2D(im, -1, k, k)
    mu_a, mu_b = blur(a), blur(b)
    sa = blur(a * a) - mu_a ** 2
    sb = blur(b * b) - mu_b ** 2
    sab = blur(a * b) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return float((((2 * mu_a * mu_b + c1) * (2 * sab + c2)) /
                  ((mu_a ** 2 + mu_b ** 2 + c1) * (sa + sb + c2))).mean())


def flow_warp(prev, prev_gray, cur_gray):
    flow = cv2.calcOpticalFlowFarneback(prev_gray, cur_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    h, w = cur_gray.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(prev, gx + flow[..., 0], gy + flow[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default=os.path.join(root, 'research/2.reproduction/MobileIE'))
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

    # 先生成整段输入与模型输出（一次推理，供所有方法复用）
    rng = np.random.default_rng(0)
    gts, ins, outs = [], [], []
    for t in range(N_FRAMES):
        gt = frame_at(img, t)
        inp = degrade(gt, rng)
        gts.append(gt); ins.append(inp); outs.append(enhance(net, inp, device))
        if (t + 1) % 30 == 0:
            print('inference %d/%d' % (t + 1, N_FRAMES))

    def evaluate(method, alpha):
        smooth_prev = None
        flick, ps, ss = [], [], []
        for t in range(N_FRAMES):
            out = outs[t].astype(np.float32)
            if alpha > 0 and smooth_prev is not None:
                if method == 'flow_ema':
                    base = flow_warp(smooth_prev, cv2.cvtColor(ins[t - 1], cv2.COLOR_BGR2GRAY),
                                     cv2.cvtColor(ins[t], cv2.COLOR_BGR2GRAY))
                else:
                    base = smooth_prev
                sm = alpha * base + (1 - alpha) * out
            else:
                sm = out
            sm_u8 = np.clip(sm, 0, 255).astype(np.uint8)
            smooth_prev = sm
            if t > 0:
                flick.append(aligned_diff(prev_sm, sm_u8))
            prev_sm = sm_u8
            ps.append(psnr(sm_u8, gts[t])); ss.append(ssim(sm_u8, gts[t]))
        return {'alpha': alpha, 'flicker': float(np.mean(flick)),
                'psnr': float(np.mean(ps)), 'ssim': float(np.mean(ss))}

    results = {'none': evaluate('none', 0.0)}
    for method in ('ema', 'flow_ema'):
        results[method] = [evaluate(method, a) for a in ALPHAS if a > 0]
    results['input'] = {
        'flicker': float(np.mean([aligned_diff(ins[t - 1], ins[t]) for t in range(1, N_FRAMES)])),
        'psnr': float(np.mean([psnr(ins[t], gts[t]) for t in range(N_FRAMES)])),
        'ssim': float(np.mean([ssim(ins[t], gts[t]) for t in range(N_FRAMES)])),
    }

    out_dir = os.path.join(root, OUT)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(results, open(os.path.join(out_dir, 'baselines.json'), 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for method, label, color in (('ema', 'EMA（无运动补偿）', '#2f6feb'),
                                 ('flow_ema', '光流对齐 EMA', '#1a7f37')):
        xs = [r['flicker'] for r in results[method]]
        ys = [r['psnr'] for r in results[method]]
        ax.plot(xs, ys, 'o-', label=label, color=color)
        for r in results[method]:
            ax.annotate('α=%.1f' % r['alpha'], (r['flicker'], r['psnr']),
                        textcoords='offset points', xytext=(5, 4), fontsize=8)
    n = results['none']
    ax.scatter([n['flicker']], [n['psnr']], marker='*', s=180, color='#bc4c00', label='MobileIE（无平滑）')
    ax.set_xlabel('闪烁（对齐帧间残差，越低越好）')
    ax.set_ylabel('PSNR / dB（越高越好）')
    ax.set_title('Training-free 平滑：闪烁与画质的权衡')
    ax.grid(alpha=.3); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'tradeoff.png')); plt.close(fig)

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
