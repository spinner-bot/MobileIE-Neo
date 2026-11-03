"""生成汇报用图表：逐图指标、闪烁对比柱状图、局部放大对比图。"""
import json
import os
import re
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 130


# ---------- 1. LOLv1 eval15 逐图指标 ----------
def fig_eval():
    log = os.path.join(ROOT, 'research/2.reproduction/results/eval_lolv1_eval15.log')
    names, ps, ss = [], [], []
    for line in open(log, encoding='utf-8'):
        m = re.match(r'(\S+)\.png\s+PSNR\s+([\d.]+)\s+SSIM\s+([\d.]+)', line.strip())
        if m:
            names.append(m.group(1)); ps.append(float(m.group(2))); ss.append(float(m.group(3)))
    if not names:
        print('eval log not found'); return
    order = np.argsort(ps)
    names = [names[i] for i in order]; ps = [ps[i] for i in order]; ss = [ss[i] for i in order]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].bar(names, ps, color='#2f6feb', alpha=.85)
    axes[0].axhline(np.mean(ps), color='#bc4c00', ls='--', lw=1.2,
                    label='mean %.3f dB' % np.mean(ps))
    axes[0].set_ylabel('PSNR (dB)'); axes[0].legend(); axes[0].grid(axis='y', alpha=.3)
    axes[1].bar(names, ss, color='#1a7f37', alpha=.85)
    axes[1].axhline(np.mean(ss), color='#bc4c00', ls='--', lw=1.2,
                    label='mean %.4f' % np.mean(ss))
    axes[1].set_ylabel('SSIM'); axes[1].legend(); axes[1].grid(axis='y', alpha=.3)
    for ax in axes:
        ax.tick_params(axis='x', rotation=60)
    fig.suptitle('MobileIE on LOLv1 eval15 (15 images)')
    fig.tight_layout()
    out = os.path.join(ROOT, 'research/2.reproduction/results/per_image_metrics.png')
    fig.savefig(out); plt.close(fig)
    print('wrote', out)


# ---------- 2. 闪烁对比柱状图 ----------
def fig_flicker():
    m = json.load(open(os.path.join(ROOT, 'research/3.video-flicker/results_realistic/metrics.json'),
                       encoding='utf-8'))
    groups = ['clean', 'realistic', 'frozen_att']
    labels = ['仅低光\n(clean)', '噪声+压缩\n(realistic)', '冻结全局注意力\n(ablation)']
    inp = [m[g]['input_aligned_diff'] for g in groups]
    out = [m[g]['output_aligned_diff'] for g in groups]

    x = np.arange(len(groups)); w = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.4))
    b1 = ax.bar(x - w / 2, inp, w, label='输入视频', color='#8b96a5')
    b2 = ax.bar(x + w / 2, out, w, label='MobileIE 输出', color='#bc4c00')
    ax.set_yscale('log')
    ax.set_ylabel('帧间对齐残差 (0-255，对数轴)')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title('时序不一致：同一场景点相邻帧差异（越低越稳定）')
    ax.grid(axis='y', alpha=.3, which='both')
    ax.legend()
    for bars in (b1, b2):
        for b in bars:
            ax.annotate('%.2f' % b.get_height(), (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha='center', va='bottom', fontsize=9)
    for i, g in enumerate(groups):
        ax.annotate('放大 %.2f×' % m[g]['amplification'], (i, max(inp[i], out[i]) * 2.2),
                    ha='center', fontsize=9, color='#bc4c00')
    fig.tight_layout()
    out_path = os.path.join(ROOT, 'research/3.video-flicker/results_realistic/flicker_bars.png')
    fig.savefig(out_path); plt.close(fig)
    print('wrote', out_path)


# ---------- 3. 局部放大对比 ----------
def extract_frame(video, idx, dst):
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', video,
                    '-vf', 'select=eq(n\\,%d)' % idx, '-vframes', '1', dst], check=True)


def fig_crop():
    d = os.path.join(ROOT, 'research/3.video-flicker/results_realistic')
    tmp = 'F:/pip-tmp'
    pairs = [(os.path.join(d, 'input_lowlight.mp4'), 'input.png'),
             (os.path.join(d, 'output_mobileie.mp4'), 'output.png'),
             (os.path.join(d, 'ground_truth.mp4'), 'gt.png')]
    for video, name in pairs:
        extract_frame(video, 30, os.path.join(tmp, name))

    box = (120, 180, 380, 375)   # 纹理区域
    titles = ['输入（噪声 + 压缩）', 'MobileIE 输出', '参考 GT']
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, (_, name), title in zip(axes, pairs, titles):
        im = Image.open(os.path.join(tmp, name)).crop(box)
        im = im.resize((im.width * 2, im.height * 2), Image.NEAREST)
        ax.imshow(np.array(im)); ax.set_title(title, fontsize=12); ax.axis('off')
    fig.suptitle('同一场景局部放大（第 31 帧）', fontsize=13)
    fig.tight_layout()
    out = os.path.join(d, 'crop_compare.png')
    fig.savefig(out); plt.close(fig)
    print('wrote', out)


if __name__ == '__main__':
    fig_eval()
    fig_flicker()
    fig_crop()
