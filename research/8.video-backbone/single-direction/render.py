"""All S1 means and all-window last-frame contacts; no new inference or scores."""
from pathlib import Path
import csv
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sys.path.insert(0, str(ROOT / 'research/4.temporal-consistency/scripts'))
from causal_reference_io_r75 import write, digest, stamp


def main():
    source, out = BASE / 'results/run001', BASE / 'report'
    out.mkdir(exist_ok=False)
    summary = json.loads((source / 'summary.json').read_text())
    rows = [json.loads(l) for l in (source / 'per_sequence.jsonl').read_text().splitlines()]
    manifest = [json.loads(l) for l in (source / 'prediction_manifest.jsonl').read_text().splitlines()]
    details = [json.loads(l) for l in (source / 'transport_details.jsonl').read_text().splitlines()]
    means = summary['groups']
    assert len(means) == 20 and len(rows) == 80 and len(manifest) == 16 and len(details) == 48
    with (out / 'points.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(means[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(means)
    styles = {'raw': ('F0 raw', 'o', '#333333'),
              'original_FB': ('native DIS + FB', 's', '#009E73'),
              'backward_only': ('backward only', '^', '#0072B2'),
              'push_average': ('push average', 'P', '#E69F00'),
              'push_dispersion': ('push dispersion', 'D', '#CC79A7')}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (domain, seed) in zip(axes.flat, [('SYN', 0), ('SYN', 1), ('ND', 0), ('ND', 1)]):
        for arm, (label, marker, color) in styles.items():
            row = next(r for r in means if (r['seed'], r['domain'], r['arm']) == (seed, domain, arm))
            ax.scatter(row['temporal_error'], row['psnr_rgb'], label=label, marker=marker, color=color, s=65)
        ax.set_title(f'{domain} training windows / F0 seed {seed}')
        ax.set_xlabel('Common GT-derived residual temporal MAE (lower better)')
        ax.set_ylabel('RGB PSNR, dB (higher better)')
        ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3)
    fig.suptitle('S1: all 20 group means; frozen F0; fixed four-frame training windows')
    fig.tight_layout(rect=(0, .09, 1, .96))
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'quality-vs-time.{ext}', dpi=140)
    plt.close(fig)
    columns = ('gt', 'low', *styles)
    contact = Image.new('RGB', (128 * len(columns), 24 + 16 * 151), 'white')
    draw = ImageDraw.Draw(contact)
    for col, name in enumerate(columns):
        draw.text((col * 128 + 2, 4), name, fill='black')
    for index, record in enumerate(manifest):
        path = ROOT / record['path']
        assert path.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo') and digest(path) == record['sha256']
        top = 24 + index * 151
        draw.text((2, top + 3), record['key'] + ' / ' + record['domain'] + ' / t3 / all-window post-hoc contact', fill='black')
        with np.load(path) as archive:
            for col, name in enumerate(columns):
                value = archive[name][3].clip(0, 255).round().astype(np.uint8)
                contact.paste(Image.fromarray(value), (col * 128, top + 21))
    contact.save(out / 'all-windows-t3.png')
    analysis = dict(condition_counts=[], transport_acceptance=[])
    for reference in ('original_FB', 'backward_only', 'push_average', 'raw'):
        candidate = {r['key']: r for r in rows if r['arm'] == 'push_dispersion'}
        refs = {r['key']: r for r in rows if r['arm'] == reference}
        analysis['condition_counts'].append(dict(reference=reference, windows=16,
            PSNR_better=sum(candidate[k]['psnr_rgb'] > refs[k]['psnr_rgb'] for k in candidate),
            time_better=sum(candidate[k]['temporal_error'] < refs[k]['temporal_error'] for k in candidate),
            both_better=sum(candidate[k]['psnr_rgb'] > refs[k]['psnr_rgb'] and candidate[k]['temporal_error'] < refs[k]['temporal_error'] for k in candidate)))
    for mean in (r for r in means if r['arm'] == 'raw'):
        keys = {r['key'] for r in manifest if r['seed'] == mean['seed'] and r['domain'] == mean['domain']}
        selected = [r for r in details if r['key'] in keys]
        analysis['transport_acceptance'].append(dict(seed=mean['seed'], domain=mean['domain'], pairs=len(selected),
            **{k: float(np.mean([r[k] for r in selected])) for k in ('native_FB_accepted', 'backward_bounds', 'push_covered', 'push_accepted')},
            total_covered_pixels_rejected_by_dispersion=sum(r['covered_rejected_by_dispersion'] for r in selected)))
    write(out / 'descriptive-analysis.json', analysis)
    write(out / 'render.json', dict(at=stamp(), new_NN_images=0, new_metric_scores=0, threshold_changes=0,
        actual_viewed=False, groups=20, contacts='all 16 windows, frame3; post-hoc, no outcome selection',
        source_sha256={p.name: digest(p) for p in source.iterdir() if p.name in ('summary.json', 'per_sequence.jsonl', 'prediction_manifest.jsonl', 'transport_details.jsonl')},
        script_sha256=digest(__file__), artifacts=[dict(path=str(p.relative_to(ROOT)), sha256=digest(p)) for p in sorted(out.iterdir())],
        scope='Repeated training windows; not physical-flow truth, playback, MOS, independent final or phone speed'))


if __name__ == '__main__':
    main()
