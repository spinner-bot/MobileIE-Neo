"""闪烁可见性与频谱分析（基于已知平移的合成夜景视频）。

输入为 flicker_realistic.py 保存的帧序列（默认 F:/pip-tmp/mobileie_frames_real），
输出 results/visibility/ 下的 summary.json 与图表。

与 3 号工作的区别：3 号只给逐帧平均残差；本脚本进一步回答
"18/255 的残差为什么肉眼不一定明显 / 在哪些区域最明显"：

1. 对齐帧差的像素级分布（均值/中位数/分位数，超过 JND 阈值的像素比例）；
2. 平坦区域 vs 纹理区域的残差（平坦区抖动更易被察觉）；
3. 逐像素时间标准差（对齐到同一视角后），以及输入→输出的时间噪声放大；
4. 帧级残差序列的频谱与一阶自相关（区分随机抖动与结构性脉动）。

用法：
    python research/4.temporal-consistency/scripts/flicker_visibility.py
"""
import json
import os

import cv2
import numpy as np

W, H, PAN = 640, 480, 4
FPS = 30
JND = [2, 4, 8, 16, 32]          # 灰度级阈值（启发式，非心理物理测量）
FRAME_DIR = 'F:/pip-tmp/mobileie_frames_real'
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
OUT = os.path.join(ROOT, 'research/4.temporal-consistency/results/visibility')
METRICS = os.path.join(ROOT, 'research/4.temporal-consistency/results/rerun/metrics.json')


def load_frames(kind):
    d = os.path.join(FRAME_DIR, kind)
    names = sorted(n for n in os.listdir(d) if n.lower().endswith('.jpg'))
    frames = [cv2.imread(os.path.join(d, n)).astype(np.float32) for n in names]
    return frames


def aligned_diff_maps(frames):
    """相邻帧按已知平移对齐后的逐像素绝对差（通道平均）。"""
    maps = []
    for t in range(1, len(frames)):
        cur = np.roll(frames[t], PAN, axis=1)[:, :W - PAN]
        prev = frames[t - 1][:, :W - PAN]
        maps.append(np.abs(cur - prev).mean(axis=2))
    return maps


def temporal_stack(frames):
    """把所有帧对齐到第 0 帧视角，返回公共有效区域的堆叠。"""
    n = len(frames)
    x0 = PAN * (n - 1)
    stack = [np.roll(f, PAN * t, axis=1)[:, x0:] for t, f in enumerate(frames)]
    return np.stack(stack)                       # n × H × (W-x0) × 3


def texture_mask(frames):
    """用时间平均梯度把像素分为平坦区 / 纹理区。"""
    grads = []
    for f in frames:
        g = cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        grads.append(np.hypot(gx, gy) * 8)       # Sobel 缩放，回到近似灰度级
    grad = np.mean(grads, axis=0)
    thr_flat = np.median(grad)
    thr_tex = np.percentile(grad, 75)
    return grad, grad <= thr_flat, grad >= thr_tex


def dist_stats(maps):
    x = np.concatenate([m.ravel() for m in maps])
    out = {
        'mean': float(x.mean()),
        'median': float(np.median(x)),
        'p90': float(np.percentile(x, 90)),
        'p95': float(np.percentile(x, 95)),
        'p99': float(np.percentile(x, 99)),
        'max': float(x.max()),
    }
    for th in JND:
        out['frac_gt_%d' % th] = float((x > th).mean())
    return out


def region_means(maps, flat, tex):
    m = np.mean(maps, axis=0)
    flat, tex = flat[:, :m.shape[1]], tex[:, :m.shape[1]]   # 对齐帧差比原图窄 PAN 像素
    return {'flat': float(m[flat].mean()), 'texture': float(m[tex].mean())}


def temporal_std(frames):
    stack = temporal_stack(frames).mean(axis=3)  # 灰度近似
    std = stack.std(axis=0)
    return std


def spectrum(series):
    x = np.asarray(series, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    w = np.hanning(n)
    X = np.fft.rfft(x * w)
    freqs = np.fft.rfftfreq(n, 1.0 / FPS)
    psd = (np.abs(X) ** 2) / max(n, 1)
    k = int(np.argmax(psd[1:])) + 1 if len(psd) > 1 else 0
    return {
        'freqs': freqs.tolist(),
        'psd': psd.tolist(),
        'dominant_hz': float(freqs[k]),
        'total_power': float(psd.sum()),
        'lag1_autocorr': float(np.corrcoef(x[:-1], x[1:])[0, 1]) if n > 2 else 0.0,
    }


def temporal_psd(frames):
    """对齐后的逐像素时间序列平均功率谱：只保留闪烁/噪声，去掉场景运动。"""
    stack = temporal_stack(frames).mean(axis=3)
    x = stack - stack.mean(axis=0, keepdims=True)
    n, h, w = x.shape
    win = np.hanning(n)[:, None, None]
    X = np.fft.rfft(x * win, axis=0)
    psd = (np.abs(X) ** 2) / max(n, 1)
    mean_psd = psd.mean(axis=(1, 2))
    freqs = np.fft.rfftfreq(n, 1.0 / FPS)
    pos = freqs > 0
    tot = mean_psd[pos].sum()
    bands = {
        'lt1hz': float(mean_psd[pos & (freqs <= 1)].sum() / max(tot, 1e-12)),
        '1to3hz': float(mean_psd[pos & (freqs > 1) & (freqs <= 3)].sum() / max(tot, 1e-12)),
        'gt3hz': float(mean_psd[pos & (freqs > 3)].sum() / max(tot, 1e-12)),
    }
    k = int(np.argmax(mean_psd[1:])) + 1 if len(mean_psd) > 1 else 0
    return {'freqs': freqs.tolist(), 'psd': mean_psd.tolist(),
            'dominant_hz': float(freqs[k]), 'bands': bands,
            'total_power': float(mean_psd[pos].sum())}


def plot_all(series, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 1) 帧级对齐残差序列
    fig, ax = plt.subplots(figsize=(10, 3.8))
    t = np.arange(1, len(series['output']['series']) + 1)
    ax.plot(t, series['input']['series'], label='输入（噪声 + JPEG）', lw=1.1, color='#8c959f')
    ax.plot(t, series['output']['series'], label='MobileIE 输出', lw=1.3, color='#bc4c00')
    for th, c in zip([4, 8, 16], ['#cfd8dc', '#b0bec5', '#90a4ae']):
        ax.axhline(th, color=c, ls=':', lw=1)
        ax.text(1.0, th, ' %d/255' % th, va='bottom', fontsize=8, color='#607d8b')
    ax.set_xlabel('帧序号'); ax.set_ylabel('对齐帧差（0-255）')
    ax.set_title('帧级对齐残差：输入 vs MobileIE 输出')
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'series.png'), dpi=140); plt.close(fig)

    # 2) 像素级残差分布
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for name, color, maps in series['_maps']:
        x = np.concatenate([m.ravel() for m in maps])
        ax.hist(x, bins=np.linspace(0, 80, 161), density=True,
                histtype='step', lw=1.4, color=color, label=name)
    for th in [4, 8, 16, 32]:
        ax.axvline(th, color='#90a4ae', ls=':', lw=1)
        ax.text(th, ax.get_ylim()[1] * 0.9, ' %d' % th, fontsize=8, color='#607d8b')
    ax.set_yscale('log'); ax.set_xlabel('像素对齐帧差（0-255）'); ax.set_ylabel('概率密度（log）')
    ax.set_title('像素级残差分布（虚线为 JND 启发式阈值）')
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'hist.png'), dpi=140); plt.close(fig)

    # 3) 时间标准差与放大
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    vmax = np.percentile(series['_std_out'], 99)
    im0 = axes[0].imshow(series['_std_out'], cmap='inferno', vmin=0, vmax=vmax)
    axes[0].set_title('输出逐像素时间标准差（对齐后）'); axes[0].axis('off')
    fig.colorbar(im0, ax=axes[0], fraction=.046)
    amp = np.clip(series['_std_out'] / (series['_std_in'] + 1e-3), 0, 30)
    im1 = axes[1].imshow(amp, cmap='magma', vmin=0, vmax=np.percentile(amp, 99))
    axes[1].set_title('时间噪声放大倍数（输出 / 输入）'); axes[1].axis('off')
    fig.colorbar(im1, ax=axes[1], fraction=.046)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'temporal_std_maps.png'), dpi=140); plt.close(fig)

    # 4) 频谱（对齐后的逐像素时间序列，去除了场景运动）
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(series['temporal_spectrum']['input']['freqs'][1:],
            series['temporal_spectrum']['input']['psd'][1:], label='输入', color='#8c959f')
    ax.plot(series['temporal_spectrum']['output']['freqs'][1:],
            series['temporal_spectrum']['output']['psd'][1:], label='MobileIE 输出', color='#bc4c00')
    ax.set_yscale('log'); ax.set_xlabel('时间频率（Hz）'); ax.set_ylabel('平均功率谱密度（log）')
    ax.set_title('逐像素时间噪声频谱（对齐后）'); ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'spectrum.png'), dpi=140); plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    inp = load_frames('input')
    out = load_frames('output')
    gt = load_frames('ground_truth')
    n = len(inp)
    print('frames:', n)

    maps_in = aligned_diff_maps(inp)
    maps_out = aligned_diff_maps(out)
    grad, flat, tex = texture_mask(gt)

    std_in = temporal_std(inp)
    std_out = temporal_std(out)
    # 纹理掩码裁剪到时间堆叠后的公共区域
    x0 = PAN * (n - 1)
    flat_c, tex_c = flat[:, x0:], tex[:, x0:]

    metrics = json.load(open(METRICS, encoding='utf-8')) if os.path.isfile(METRICS) else {}
    series_out = metrics.get('realistic', {}).get('per_frame_aligned') or np.mean(maps_out, axis=(1, 2)).tolist()
    series_in = metrics.get('realistic', {}).get('per_frame_input_aligned') or np.mean(maps_in, axis=(1, 2)).tolist()

    summary = {
        'frames': n,
        'jnd_thresholds': JND,
        'pixel_diff': {'input': dist_stats(maps_in), 'output': dist_stats(maps_out)},
        'region_diff': {'input': region_means(maps_in, flat, tex),
                        'output': region_means(maps_out, flat, tex)},
        'temporal_std': {
            'input_mean': float(std_in.mean()), 'input_p95': float(np.percentile(std_in, 95)),
            'output_mean': float(std_out.mean()), 'output_p95': float(np.percentile(std_out, 95)),
            'amplification_mean': float(std_out.mean() / max(std_in.mean(), 1e-6)),
            'flat_input': float(std_in[flat_c].mean()), 'flat_output': float(std_out[flat_c].mean()),
            'texture_input': float(std_in[tex_c].mean()), 'texture_output': float(std_out[tex_c].mean()),
        },
        'frame_series': {
            'input': spectrum(series_in),
            'output': spectrum(series_out),
        },
        'temporal_spectrum': {
            'input': temporal_psd(inp),
            'output': temporal_psd(out),
        },
    }
    summary['_maps'] = [('输入', '#8c959f', maps_in), ('MobileIE 输出', '#bc4c00', maps_out)]
    summary['_std_in'], summary['_std_out'] = std_in, std_out
    summary['input'] = {'series': series_in, **summary['frame_series']['input']}
    summary['output'] = {'series': series_out, **summary['frame_series']['output']}

    plot_all(summary, OUT)
    for k in ('_maps', '_std_in', '_std_out', 'input', 'output'):
        summary.pop(k, None)
    with open(os.path.join(OUT, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
