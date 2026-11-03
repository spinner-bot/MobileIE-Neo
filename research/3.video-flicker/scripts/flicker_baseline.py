"""MobileIE 视频闪烁基线实验。

构造一段已知运动（匀速平移）的合成低光视频，逐帧用 MobileIE 增强，
量化帧间不一致（闪烁）：输入视频与输出视频的时序差异放大倍数、对齐残差。

用法（任意目录运行）：
    python research/3.video-flicker/scripts/flicker_baseline.py

内存友好：逐帧流式处理，不同时持有所有帧。
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

W, H = 640, 480          # 视频分辨率
MOVE = 2                 # 每帧平移像素（视频内实际平移 = 2*MOVE）
PAN = MOVE * 2            # 每帧真实平移像素
N_FRAMES = 60
FPS = 30


def frame_pair(img_bgr, t):
    """第 t 帧：GT 图 + 模拟低光输入图。"""
    x = MOVE * 2 * t
    x = min(x, img_bgr.shape[1] - W)
    y0 = (img_bgr.shape[0] - H) // 2
    gt = img_bgr[y0:y0 + H, x:x + W]
    low = np.clip(((gt.astype(np.float32) / 255.0) ** 2.2) * 0.35, 0, 1)
    return gt, (low * 255).astype(np.uint8)


def to_tensor(bgr_uint8, device):
    rgb = cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)


def enhance_one(net, frame_bgr, device):
    with torch.no_grad():
        out = net(to_tensor(frame_bgr, device)).clamp(0, 1)[0]
    arr = (out.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    del out
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def aligned_diff(prev, cur, shift=PAN):
    """cur 反向平移 shift 像素后与 prev 比较（已知匀速平移的精确对齐）。"""
    # 画面整体左移 shift 像素，把 cur 右移 shift 后与 prev 对齐
    cur_aligned = np.roll(cur, shift, axis=1)
    a = prev[:, :W - shift].astype(np.float32)
    b = cur_aligned[:, :W - shift].astype(np.float32)
    return float(np.abs(a - b).mean())


def raw_diff(prev, cur):
    return float(np.abs(prev.astype(np.float32) - cur.astype(np.float32)).mean())


# 帧与中间视频都放在 F 盘 ASCII 路径：ffmpeg 无法写中文路径，C 盘空间紧张
FRAME_DIR = 'F:/pip-tmp/mobileie_frames'


def save_frame(kind, idx, frame):
    """逐帧存 jpg，最后用 ffmpeg 合成视频（cv2.VideoWriter 在本机写入失败）。"""
    d = os.path.join(FRAME_DIR, kind)
    os.makedirs(d, exist_ok=True)
    ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if ok:
        buf.tofile(os.path.join(d, '%04d.jpg' % idx))


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default=os.path.join(root, 'research/2.reproduction/MobileIE'))
    ap.add_argument('--image', default=os.path.join(root, 'research/3.video-flicker/assets/source_scene.jpg'))
    ap.add_argument('--weights', default='pretrain/lolv1_best_slim.pkl')
    ap.add_argument('--out', default=os.path.join(root, 'research/3.video-flicker/results'))
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    repo = os.path.abspath(args.repo)
    sys.path.insert(0, repo)
    os.chdir(repo)
    from model.lle import MobileIELLENetS

    use_cuda = args.device.startswith('cuda') and torch.cuda.is_available()
    device = torch.device(args.device if use_cuda else 'cpu')
    print('device:', device)
    net = MobileIELLENetS(channels=12)
    net.load_state_dict(torch.load(args.weights, map_location='cpu', weights_only=True))
    net.eval().to(device)

    img_bgr = cv2.cvtColor(np.array(Image.open(args.image).convert('RGB')), cv2.COLOR_RGB2BGR)
    img_bgr = cv2.resize(img_bgr, (1600, 1200), interpolation=cv2.INTER_AREA)

    os.makedirs(args.out, exist_ok=True)
    files = {'input': 'input_lowlight.mp4', 'output': 'output_mobileie.mp4', 'ground_truth': 'ground_truth.mp4'}
    import shutil
    shutil.rmtree(FRAME_DIR, ignore_errors=True)

    stats = {k: {'raw': [], 'aligned': []} for k in ('input', 'output')}
    prev = {}
    for t in range(N_FRAMES):
        gt, low = frame_pair(img_bgr, t)
        out = enhance_one(net, low, device)
        for name, frame in (('input', low), ('output', out), ('ground_truth', gt)):
            save_frame(name, t, frame)
        if t > 0:
            for name, cur in (('input', low), ('output', out)):
                stats[name]['raw'].append(raw_diff(prev[name], cur))
                stats[name]['aligned'].append(aligned_diff(prev[name], cur))
        prev = {'input': low, 'output': out}
        if (t + 1) % 20 == 0:
            print(f'frame {t + 1}/{N_FRAMES}')

    # 视频合成交给独立进程 scripts/make_videos.py：
    # 本机内存仅 15 GB，torch/CUDA 进程内 x264 malloc 失败，需在干净的进程里编码。
    with open(os.path.join(args.out, 'video_jobs.json'), 'w', encoding='utf-8') as f:
        json.dump({'frame_dir': FRAME_DIR.replace('\\', '/'), 'out_dir': args.out.replace('\\', '/'),
                   'fps': FPS, 'files': files}, f, indent=2)

    mean = lambda xs: float(np.mean(xs)) if xs else 0.0
    report = {
        'frames': N_FRAMES, 'resolution': [W, H], 'pan_px_per_frame': MOVE, 'weights': args.weights,
        'input': {'raw_diff_mean': mean(stats['input']['raw']),
                  'aligned_diff_mean': mean(stats['input']['aligned'])},
        'output': {'raw_diff_mean': mean(stats['output']['raw']),
                   'aligned_diff_mean': mean(stats['output']['aligned'])},
        'amplification_raw': mean(stats['output']['raw']) / max(mean(stats['input']['raw']), 1e-8),
        'amplification_aligned': mean(stats['output']['aligned']) / max(mean(stats['input']['aligned']), 1e-8),
        'per_frame_aligned_input': stats['input']['aligned'],
        'per_frame_aligned_output': stats['output']['aligned'],
    }
    with open(os.path.join(args.out, 'flicker_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        t = np.arange(1, N_FRAMES)
        plt.figure(figsize=(10, 4))
        plt.plot(t, stats['input']['aligned'], label='input (aligned)')
        plt.plot(t, stats['output']['aligned'], label='MobileIE output (aligned)')
        plt.xlabel('frame index')
        plt.ylabel('aligned mean abs diff')
        plt.title('Temporal instability per frame')
        plt.legend()
        plt.grid(alpha=.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, 'flicker_curve.png'), dpi=130)
        plt.close()
    except Exception as e:
        print('plot skipped:', e)

    print(json.dumps({k: v for k, v in report.items() if not k.startswith('per_frame')},
                     indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
