"""Post-hoc complete P1 plots and fixed t4/photons20/rep0 contacts; zero new NN."""
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
from causal_reference_io_r75 import write, stamp, digest


def main():
    source = BASE / 'results/run001'
    summary = json.loads((source / 'summary.json').read_text())
    rows = [json.loads(line) for line in (source / 'per_sequence.jsonl').read_text().splitlines()]
    manifest = [json.loads(line) for line in (source / 'prediction_manifest.jsonl').read_text().splitlines()]
    out = BASE / 'report'
    out.mkdir(exist_ok=False)
    means = summary['groups']
    assert len(means) == 30 and len(rows) == 200 and len(manifest) == 40
    with (out / 'points.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(means[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(means)
    labels = {'raw': ('F0 raw', 'o', '#333333'),
              'feature_ema': ('feature EMA', 's', '#0072B2'),
              'envelope_spatial': ('spatial envelope', '^', '#E69F00'),
              'envelope_temporal': ('temporal envelope', 'D', '#CC79A7'),
              'known_motion_post': ('known-motion RGB post', '*', '#009E73')}
    fig, axes = plt.subplots(3, 2, figsize=(11, 12))
    for row_index, photons in enumerate((None, 20, 80)):
        for seed in (0, 1):
            ax = axes[row_index, seed]
            for arm, (label, marker, color) in labels.items():
                row = next(r for r in means if (r['seed'], r['photons'], r['arm']) == (seed, photons, arm))
                ax.scatter(row['temporal_MSE'], row['PSNR'], color=color, marker=marker, s=65, label=label)
            ax.set_title(f'{"Clean quantized degradation" if photons is None else "Photons " + str(photons)} / F0 seed {seed}')
            ax.set_xlabel('Known-motion GT-error temporal MSE (lower better)')
            ax.set_ylabel('Late-frame PSNR, dB (higher better)')
            ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
            ax.grid(alpha=.2)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc='lower center', ncol=3)
    fig.suptitle('P1 frozen-backbone screening: all 30 group means; training textures only')
    fig.tight_layout(rect=(0, .065, 1, .965))
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'quality-vs-time.{ext}', dpi=140)
    plt.close(fig)
    columns = ('gt', 'low', *labels)
    contact = Image.new('RGB', (128 * len(columns), 24 + 8 * 151), 'white')
    draw = ImageDraw.Draw(contact)
    for col, name in enumerate(columns):
        draw.text((col * 128 + 3, 4), name.replace('envelope_', 'env_').replace('known_motion_post', 'known RGB'), fill='black')
    for seed in (0, 1):
        for texture in range(4):
            entry = next(r for r in manifest if r['seed'] == seed and r['key'] == f'texture{texture}-p20-rep0')
            path = ROOT / entry['path']
            assert path.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo') and digest(path) == entry['sha256']
            index = seed * 4 + texture
            top = 24 + index * 151
            draw.text((3, top + 2), f'seed{seed} texture{texture} | photons20 rep0 t4 | post-hoc static contact', fill='black')
            with np.load(path) as archive:
                for col, name in enumerate(columns):
                    image = (archive[name][4].transpose(1, 2, 0).clip(0, 1) * 255).round().astype(np.uint8)
                    contact.paste(Image.fromarray(image), (col * 128, top + 21))
    contact.save(out / 'contacts-t4-p20-rep0.png')
    comparisons = []
    for reference in ('raw', 'envelope_spatial', 'feature_ema', 'known_motion_post'):
        for clean in (False, True):
            candidate = {(r['seed'], r['key']): r for r in rows if r['arm'] == 'envelope_temporal' and (r['photons'] is None) == clean}
            refs = {(r['seed'], r['key']): r for r in rows if r['arm'] == reference and (r['photons'] is None) == clean}
            assert candidate.keys() == refs.keys()
            comparisons.append(dict(reference=reference, clean=clean, conditions=len(candidate),
                PSNR_better=sum(candidate[k]['PSNR'] > refs[k]['PSNR'] for k in candidate),
                time_better=sum(candidate[k]['temporal_MSE'] < refs[k]['temporal_MSE'] for k in candidate),
                both_better=sum(candidate[k]['PSNR'] > refs[k]['PSNR'] and candidate[k]['temporal_MSE'] < refs[k]['temporal_MSE'] for k in candidate)))
    write(out / 'condition_counts.json', comparisons)
    artifacts = [dict(path=str(p.relative_to(ROOT)), sha256=digest(p)) for p in sorted(out.iterdir())]
    write(out / 'render.json', dict(at=stamp(), all_points=30, sequence_metrics=200, artifacts=artifacts,
          new_NN_images=0, new_scores=0, threshold_changes=0, actual_viewed=False,
          source_sha256={name: digest(source / name) for name in ('summary.json', 'per_sequence.jsonl', 'prediction_manifest.jsonl')},
          script_sha256=digest(__file__),
          contact_selection='post-hoc all four textures / both F0 seeds / photons20 rep0 t4; no outcome selection',
          evidence='training-texture controlled proxy; not playback, MOS, natural motion, phone or independent final'))


if __name__ == '__main__':
    main()
