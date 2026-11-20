"""Post hoc complete spatial tables/figures; no Torch import or new inference."""
import csv
import json
import os
from pathlib import Path

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study_io import BASE, ROOT, read, write, stamp, digest, verify


def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def csv_file(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def main():
    verify()
    out = BASE / 'report/spatial-analysis'
    out.mkdir(parents=True, exist_ok=False)
    rows, frames, manifests = [], [], []
    for seed in (0, 1):
        source = BASE / f'results/spatial-held-seed{seed}'
        assert read(source / 'runtime.json')['completed_normally']
        rows += lines(source / 'per_sequence.jsonl')
        frames += lines(source / 'per_frame.jsonl')
        manifests += lines(source / 'prediction_manifest.jsonl')
    assert (len(rows), len(frames), len(manifests)) == (384, 1536, 128)
    means = read(BASE / 'report/spatial-recovery/domain_seed_means.json')
    assert len(means) == 12
    by_key = {(r['key'], r['arm']): r for r in rows}
    pairs = []
    for record in manifests:
        k = record['key']; m = by_key[k, 'multiaffine']
        pair = dict(key=k, seed=m['seed'], domain=m['domain'])
        for ref in ('F0', 'original_scratch'):
            for metric in ('psnr_rgb', 'ssim_rgb', 'edge_mae'):
                pair[f'{metric}_delta_to_{ref}'] = m[metric] - by_key[k, ref][metric]
        pairs.append(pair)
    csv_file(out / 'all-128-paired-windows.csv', pairs)
    points = []
    for r in means:
        points.append(dict(**r, spatial_conv_GMAC_1080p=(1308*1080*1920 if r['arm'] == 'multiaffine'
            else 3828*1080*1920+144)/1e9,
            initialization='random+8K' if r['arm'] != 'F0' else 'official_pretrained+project_8K_expanded'))
    csv_file(out / 'all-12-means.csv', points)
    counts = []
    for seed in (0, 1):
        for domain in ('SYN', 'ND'):
            selected = [p for p in pairs if (p['seed'], p['domain']) == (seed, domain)]
            assert len(selected) == 32
            for ref in ('F0', 'original_scratch'):
                counts.append(dict(seed=seed, domain=domain, reference=ref, windows=32,
                    PSNR_positive=sum(p[f'psnr_rgb_delta_to_{ref}'] > 0 for p in selected),
                    SSIM_positive=sum(p[f'ssim_rgb_delta_to_{ref}'] > 0 for p in selected),
                    edge_lower=sum(p[f'edge_mae_delta_to_{ref}'] < 0 for p in selected)))
    write(out / 'descriptive-counts.json', dict(at=stamp(), comparisons=counts,
        PSNR_capped_frames=sum(r['psnr_120dB_ceiling_applied'] for r in frames), frames=len(frames),
        raw_student_ranges={arm: dict(min=min(r['raw_min'] for r in rows if r['arm'] == arm),
            max=max(r['raw_max'] for r in rows if r['arm'] == arm),
            mean_outside_fraction=float(np.mean([r['outside_display_range_fraction'] for r in rows if r['arm'] == arm])))
            for arm in ('multiaffine', 'original_scratch')},
        F0_range_record_is_already_clipped=True, new_inference_or_metric_calls=0,
        counts_are_descriptive_not_independent_trials=True))
    names = dict(multiaffine='M1 random + 8K', original_scratch='Slim random + 8K', F0='F0 pretrained + project 8K')
    colors = dict(multiaffine='#c43b34', original_scratch='#387cb4', F0='#333333')
    markers = dict(multiaffine='o', original_scratch='s', F0='^')
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.5), sharex=True)
    for ax, (seed, domain) in zip(axes.flat, ((0, 'SYN'), (1, 'SYN'), (0, 'ND'), (1, 'ND'))):
        for arm in names:
            p = next(p for p in points if (p['seed'], p['domain'], p['arm']) == (seed, domain, arm))
            ax.scatter(p['spatial_conv_GMAC_1080p'], p['psnr_rgb'], c=colors[arm], marker=markers[arm], s=65, label=names[arm])
            ax.annotate(f"{p['psnr_rgb']:.2f}", (p['spatial_conv_GMAC_1080p'], p['psnr_rgb']), xytext=(0, 9),
                        textcoords='offset points', ha='center', fontsize=9)
        ax.set_title(f'{domain}, seed {seed} (32 matched training windows)')
        ax.set_xlim(1.8, 8.9); ax.margins(y=.2); ax.grid(alpha=.25)
        ax.set_ylabel('Display RGB PSNR (dB)'); ax.set_xlabel('Analytical spatial Conv GMAC at 1080p')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(.5, .055))
    fig.suptitle('M1 spatial prerequisite: all 12 means; unmatched pretraining history')
    fig.text(.5, .015, 'Training-scene draws, not independent DEV. Conv only: no latency, streaming state or full video cost.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .105, 1, .95))
    for ext in ('png', 'pdf'): fig.savefig(out / ('all-means.'+ext), dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.8, 7.3))
    for (seed, domain), color, marker in zip(((0, 'SYN'), (1, 'SYN'), (0, 'ND'), (1, 'ND')),
                                           ('#c43b34', '#dd9738', '#387cb4', '#3a9478'), ('o', 's', '^', 'D')):
        chosen = [p for p in pairs if (p['seed'], p['domain']) == (seed, domain)]
        ax.scatter([p['psnr_rgb_delta_to_F0'] for p in chosen], [p['psnr_rgb_delta_to_original_scratch'] for p in chosen],
                   label=f'{domain}, seed {seed}: all 32', color=color, marker=marker, alpha=.75, s=32)
    ax.axhline(0, color='black', lw=.8); ax.axvline(0, color='black', lw=.8)
    ax.set_xlabel('M1 minus F0 window PSNR (dB)'); ax.set_ylabel('M1 minus random Slim window PSNR (dB)')
    ax.set_title('All 128 matched windows; local gains retained, no point selection')
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout(rect=(0, .04, 1, 1))
    fig.text(.5, .015, 'Descriptive training-pool draws; paired frames/windows are not independent trials.', ha='center', fontsize=9)
    for ext in ('png', 'pdf'): fig.savefig(out / ('all-window-deltas.'+ext), dpi=180)
    plt.close(fig)
    # Deterministic post hoc illustration: first held step, first slot of each domain, last frame.
    selected_images = []
    fig, axes = plt.subplots(4, 5, figsize=(10, 8.5))
    arms = ('gt', 'low', 'F0', 'multiaffine', 'original_scratch')
    for row_index, (seed, index) in enumerate(((0, 0), (0, 2), (1, 0), (1, 2))):
        key = f'seed{seed}-step12000-window{index}'
        entry = next(m for m in manifests if m['key'] == key)
        p = ROOT / entry['path']
        assert p.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo') and digest(p) == entry['sha256']
        with np.load(p) as data:
            assert all(data[arm].shape == (4, 3, 128, 128) for arm in arms)
            for col, arm in enumerate(arms):
                ax = axes[row_index, col]
                ax.imshow(data[arm][3].clip(0, 1).transpose(1, 2, 0), vmin=0, vmax=1, interpolation='nearest')
                ax.set_xticks([]); ax.set_yticks([])
                if row_index == 0: ax.set_title(dict(gt='GT', low='Low', F0='F0', multiaffine='M1', original_scratch='Random Slim')[arm])
                if col == 0: ax.set_ylabel(f"seed {seed}, {entry['domain']}\nstep12000, slot{index}, f3", fontsize=9)
        selected_images.append(dict(**entry, frame=3))
    fig.suptitle('Fixed post hoc contact sheet; original 128px crops, display clipped')
    fig.tight_layout(rect=(0, .035, 1, .95))
    fig.text(.5, .01, 'Four predetermined slots; no best/worst search. Static views do not measure flicker or MOS.', ha='center', fontsize=9)
    for ext in ('png', 'pdf'): fig.savefig(out / ('fixed-contacts.'+ext), dpi=180)
    plt.close(fig)
    inputs = {str(p.relative_to(ROOT)): digest(p) for s in (0, 1) for p in
              ((BASE / f'results/spatial-held-seed{s}') / n for n in ('per_sequence.jsonl', 'per_frame.jsonl', 'prediction_manifest.jsonl'))}
    files = {p.name: digest(p) for p in sorted(out.iterdir())}
    write(out / 'render.json', dict(at=stamp(), post_hoc=True, source_sha256=digest(Path(__file__)),
        input_sha256=inputs, artifacts=files, selected_images=selected_images, mean_points=12, paired_windows=128,
        new_model_calls=0, new_metric_calls=0, actual_viewed=False, PDFs_exported_not_separately_viewed=True,
        inherited_source_freeze_verified=True, Torch_imported=False))
    print(json.dumps(dict(points=12, windows=len(pairs), contacts=len(selected_images), caps=sum(r['psnr_120dB_ceiling_applied'] for r in frames))))


if __name__ == '__main__':
    main()
