"""通用工具：任意视频 → MobileIE 逐帧增强 → 时序一致性指标 + 对比视频。

不假设已知运动：用 Farneback 光流把上一帧输出 warp 到当前帧，计算对齐残差；
同时给出原始帧差、亮度漂移与放大倍数。真实相机素材到手后可直接使用。

用法:
    python research/3.video-flicker/scripts/process_video.py -v path/to/video.mp4 -o out_dir
    python research/3.video-flicker/scripts/process_video.py -v video.mp4 --weights pretrain/uieb_best_slim.pkl
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np
import torch

WORK = 'F:/pip-tmp/process_video'   # ffmpeg / cv2 对中文路径不友好，中间文件放 ASCII 路径


def extract_frames(video, dst):
    os.makedirs(dst, exist_ok=True)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', video,
                    os.path.join(dst, '%05d.png')], check=True)
    return sorted(os.listdir(dst))


def imread_u(path):
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def imwrite_u(path, img):
    ok, buf = cv2.imencode('.png', img)
    if ok:
        buf.tofile(path)
    return ok


def to_tensor(bgr, device):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)


def enhance(net, bgr, device):
    with torch.no_grad():
        out = net(to_tensor(bgr, device)).clamp(0, 1)[0]
    arr = (out.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def flow_warp(prev, cur_gray, prev_gray):
    """用 prev→cur 的光流把 prev 图映射到 cur 视角。"""
    flow = cv2.calcOpticalFlowFarneback(prev_gray, cur_gray, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    h, w = cur_gray.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = gx + flow[..., 0]
    map_y = gy + flow[..., 1]
    return cv2.remap(prev, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-v', '--video', required=True)
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--weights', default='pretrain/lolv1_best_slim.pkl')
    ap.add_argument('--repo', default=os.path.join(os.path.dirname(__file__), '..', '..', '..', 'MobileIE'))
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--max-frames', type=int, default=0, help='0 表示全部')
    args = ap.parse_args()

    video_path = os.path.abspath(args.video)   # 必须在 chdir 之前解析
    out_path = os.path.abspath(args.out)
    repo = os.path.abspath(args.repo)
    sys.path.insert(0, repo)
    os.chdir(repo)
    from model.lle import MobileIELLENetS

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    net = MobileIELLENetS(channels=12)
    net.load_state_dict(torch.load(args.weights, map_location='cpu', weights_only=True))
    net.eval().to(device)

    shutil.rmtree(WORK, ignore_errors=True)
    in_dir, out_dir_frames = os.path.join(WORK, 'in'), os.path.join(WORK, 'out')
    names = extract_frames(video_path, in_dir)
    if args.max_frames:
        names = names[:args.max_frames]
    os.makedirs(out_dir_frames, exist_ok=True)

    os.makedirs(out_path, exist_ok=True)
    raw_in, raw_out, flow_in, flow_out = [], [], [], []
    prev = None
    for i, name in enumerate(names):
        inp = imread_u(os.path.join(in_dir, name))
        out = enhance(net, inp, device)
        imwrite_u(os.path.join(out_dir_frames, name), out)
        if prev is not None:
            raw_in.append(float(np.abs(inp.astype(np.float32) - prev['in'].astype(np.float32)).mean()))
            raw_out.append(float(np.abs(out.astype(np.float32) - prev['out'].astype(np.float32)).mean()))
            gi = cv2.cvtColor(prev['in'], cv2.COLOR_BGR2GRAY)
            gc = cv2.cvtColor(inp, cv2.COLOR_BGR2GRAY)
            warp_in = flow_warp(prev['in'], gc, gi)
            warp_out = flow_warp(prev['out'], gc, gi)
            flow_in.append(float(np.abs(warp_in.astype(np.float32) - inp.astype(np.float32)).mean()))
            flow_out.append(float(np.abs(warp_out.astype(np.float32) - out.astype(np.float32)).mean()))
        prev = {'in': inp, 'out': out}
        if (i + 1) % 30 == 0 or i + 1 == len(names):
            print('frame %d/%d' % (i + 1, len(names)))

    def mean(x):
        return float(np.mean(x)) if x else 0.0

    report = {
        'video': video_path,
        'frames': len(names),
        'weights': args.weights,
        'raw_diff': {'input': mean(raw_in), 'output': mean(raw_out),
                     'amplification': mean(raw_out) / max(mean(raw_in), 1e-6)},
        'flow_aligned_diff': {'input': mean(flow_in), 'output': mean(flow_out),
                              'amplification': mean(flow_out) / max(mean(flow_in), 1e-6)},
        'per_frame_flow_input': flow_in,
        'per_frame_flow_output': flow_out,
    }
    json.dump(report, open(os.path.join(out_path, 'video_metrics.json'), 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)

    # 视频合成：输入 | 输出 对比（ASCII 中间路径）
    fps = 30
    tmp_in, tmp_out, tmp_cmp = (os.path.join(WORK, n) for n in ('in.mp4', 'out.mp4', 'cmp.mp4'))
    for src, dst in ((in_dir, tmp_in), (out_dir_frames, tmp_out)):
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps),
                        '-i', os.path.join(src, '%05d.png'), '-c:v', 'libx264',
                        '-pix_fmt', 'yuv420p', '-crf', '22', '-threads', '1', dst], check=True)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', tmp_in, '-i', tmp_out,
                    '-filter_complex', '[0:v][1:v]hstack=inputs=2',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '24', '-threads', '1',
                    tmp_cmp], check=True)
    for src, name in ((tmp_in, 'input.mp4'), (tmp_out, 'output.mp4'), (tmp_cmp, 'compare_input_output.mp4')):
        shutil.move(src, os.path.join(out_path, name))

    # 曲线
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        plt.figure(figsize=(10, 4))
        t = np.arange(1, len(flow_in) + 1)
        plt.plot(t, flow_in, label='input (flow-aligned)')
        plt.plot(t, flow_out, label='MobileIE output (flow-aligned)')
        plt.xlabel('frame'); plt.ylabel('aligned mean abs diff (0-255)')
        plt.title('Temporal inconsistency (flow-aligned)')
        plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
        plt.savefig(os.path.join(out_path, 'flicker_curve.png'), dpi=130); plt.close()
    except Exception as e:
        print('plot skipped:', e)

    print(json.dumps({k: v for k, v in report.items() if not k.startswith('per_frame')},
                     indent=2, ensure_ascii=False))
    shutil.rmtree(WORK, ignore_errors=True)


if __name__ == '__main__':
    main()
