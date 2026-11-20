"""Complete B1 results, pairs and all-window contacts; no new model or metrics."""
from pathlib import Path
import sys
import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sys.path.insert(0, str(ROOT / 'research/4.temporal-consistency/scripts'))
from causal_reference_io_r75 import write, digest, stamp


def load_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def save_csv(path, rows):
    with path.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def main():
    source, out = BASE / 'results/run001', BASE / 'report'
    out.mkdir(exist_ok=False)
    summary = json.loads((source / 'summary.json').read_text())
    rows = load_lines(source / 'per_sequence.jsonl'); manifest = load_lines(source / 'prediction_manifest.jsonl')
    details = load_lines(source / 'bank-details.jsonl'); means = summary['groups']
    assert (len(means), len(rows), len(manifest), len(details)) == (20, 80, 16, 64)
    styles = dict(raw=('F0 raw', 'o', '#333333'), original_FB=('F0 + half-FB', 's', '#009E73'),
        original_DIS=('F0 + native DIS', '^', '#0072B2'), history_bank=('History bank', 'D', '#CC3355'),
        current_bank=('Current-only bank', 'P', '#D89016'))
    save_csv(out / 'all-20-means.csv', means)
    pairs = []
    by = {(r['key'], r['arm']): r for r in rows}
    counts = []
    for ref in ('raw', 'current_bank', 'original_FB', 'original_DIS'):
        for m in manifest:
            a, b = by[m['key'], 'history_bank'], by[m['key'], ref]
            pairs.append(dict(key=m['key'], seed=m['seed'], domain=m['domain'], reference=ref,
                PSNR_delta=a['psnr_rgb']-b['psnr_rgb'], SSIM_delta=a['ssim_rgb']-b['ssim_rgb'],
                edge_delta=a['edge_mae']-b['edge_mae'], Etemp_ratio=a['temporal_error']/b['temporal_error']))
        selected = [r for r in pairs if r['reference'] == ref]
        counts.append(dict(reference=ref, windows=16, PSNR_positive=sum(r['PSNR_delta'] > 0 for r in selected),
            Etemp_lower=sum(r['Etemp_ratio'] < 1 for r in selected),
            jointly_better=sum(r['PSNR_delta'] > 0 and r['Etemp_ratio'] < 1 for r in selected)))
    save_csv(out / 'all-64-paired-comparisons.csv', pairs)
    write(out / 'descriptive-analysis.json', dict(counts=counts,
        passing_comparisons=sum(r['passed'] for r in summary['comparisons']), comparisons=16,
        history_nonbootstrap_mean_acceptance=float(np.mean([r['history']['accepted_fraction'] for r in details if r['frame'] > 0])),
        history_stored_bytes=sorted({r['history']['stored_bytes'] for r in details}),
        current_stored_bytes=sorted({r['current']['stored_bytes'] for r in details}),
        mass_is_not_calibrated_match_confidence=True, independent_trials_not_claimed=True))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (domain, seed) in zip(axes.flat, [('SYN', 0), ('SYN', 1), ('ND', 0), ('ND', 1)]):
        for arm, (label, marker, color) in styles.items():
            r = next(r for r in means if (r['seed'], r['domain'], r['arm']) == (seed, domain, arm))
            ax.scatter(r['temporal_error'], r['psnr_rgb'], label=label, marker=marker, color=color, s=65)
        ax.set_title(f'{domain} / F0 seed {seed} / four training windows')
        ax.set_xlabel('GT-derived residual temporal MAE (lower better)'); ax.set_ylabel('Display RGB PSNR, dB')
        ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(.5, .025))
    fig.suptitle('B1: all 20 means; frozen original F0; no optimization')
    fig.tight_layout(rect=(0, .11, 1, .96))
    fig.text(.5, .008, 'Four-frame training draws, not independent DEV, physical-flow truth, MOS or phone timing.', ha='center', fontsize=9)
    for ext in ('png', 'pdf'): fig.savefig(out / ('quality-vs-time.'+ext), dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.8, 6.5))
    for seed, domain, color, marker in [(0, 'SYN', '#c43b34', 'o'), (1, 'SYN', '#dd9738', 's'),
                                       (0, 'ND', '#387cb4', '^'), (1, 'ND', '#3a9478', 'D')]:
        chosen = [r for r in pairs if (r['reference'], r['seed'], r['domain']) == ('current_bank', seed, domain)]
        ax.scatter([r['Etemp_ratio'] for r in chosen], [r['PSNR_delta'] for r in chosen],
            color=color, marker=marker, s=60, label=f'{domain}, seed {seed}: all four')
    ax.axvline(1, color='black', lw=.7); ax.axhline(0, color='black', lw=.7)
    ax.axvline(.95, color='gray', ls='--', label='Registered mean ratio target (.95)')
    ax.set_xlabel('History / current-only window Etemp'); ax.set_ylabel('History minus current-only window PSNR (dB)')
    ax.set_title('All 16 paired windows: history contribution beyond spatial filtering')
    ax.grid(alpha=.2); ax.legend(fontsize=9); fig.tight_layout()
    for ext in ('png', 'pdf'): fig.savefig(out / ('all-window-history-deltas.'+ext), dpi=160)
    plt.close(fig)
    columns = ('gt', 'low', *styles)
    contact = Image.new('RGB', (128*len(columns), 24+16*151), 'white')
    draw = ImageDraw.Draw(contact)
    for i, name in enumerate(columns): draw.text((i*128+2, 4), name, fill='black')
    for index, m in enumerate(manifest):
        p = ROOT / m['path']
        assert p.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo') and digest(p) == m['sha256']
        top = 24+index*151
        draw.text((2, top+3), m['key']+' / '+m['domain']+' / t3 / all-window contact', fill='black')
        with np.load(p) as data:
            for col, name in enumerate(columns):
                contact.paste(Image.fromarray(data[name][3].clip(0, 255).round().astype(np.uint8)), (col*128, top+21))
    contact.save(out / 'all-windows-t3.png')
    text = ['# B1完整固定训练窗口结果', '',
        '0优化/无新DEV；同两seed F0，所有20均值和16比较完整保留。PSNR↑、SSIM↑、edge↓、Etemp↓。', '',
        '|域|seed|输出|PSNR|SSIM|edge|Etemp|', '|---|---:|---|---:|---:|---:|---:|']
    for r in means: text.append(f"|{r['domain']}|{r['seed']}|{r['arm']}|{r['psnr_rgb']:.6f}|{r['ssim_rgb']:.6f}|{r['edge_mae']:.6f}|{r['temporal_error']:.6f}|")
    text += ['', '|域|seed|对照|PSNR差|最差窗口差|Etemp比|完整过门|', '|---|---:|---|---:|---:|---:|---|']
    for r in summary['comparisons']: text.append(f"|{r['domain']}|{r['seed']}|{r['reference']}|{r['PSNR_delta']:.6f}|{r['worst_window_PSNR_delta']:.6f}|{r['Etemp_ratio']:.6f}|{r['passed']}|")
    text += ['', 'current-only也在首帧回退raw。质量/时间均为四帧窗口，非独立长视频或感知评价；质量接纳率只是网格质量mass，不是真实对应置信度。', '']
    (out / 'tables.md').write_text('\n'.join(text))
    write(out / 'render.json', dict(at=stamp(), source_sha256=digest(__file__),
        input_sha256={p.name: digest(p) for p in source.iterdir() if p.name in ('summary.json', 'per_sequence.jsonl', 'prediction_manifest.jsonl', 'bank-details.jsonl')},
        artifacts=[dict(path=str(p.relative_to(ROOT)), sha256=digest(p)) for p in sorted(out.iterdir())],
        actual_viewed=False, post_hoc_complete_description=True, new_model_calls=0, new_metric_calls=0,
        mean_points=20, window_pairs=64, contact_windows=16, contact_frame=3, PDFs_separately_viewed=False))
    print(json.dumps(dict(means=20, pairs=len(pairs), contacts=16, passing_comparisons=sum(r['passed'] for r in summary['comparisons']))))


if __name__ == '__main__':
    main()
