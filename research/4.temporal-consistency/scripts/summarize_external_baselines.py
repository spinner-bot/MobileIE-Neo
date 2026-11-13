"""Join only completed same-base CPU external baseline stages."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

from dynamic_training_r34 import BASE, ROOT, digest

METRICS = ['psnr_rgb', 'ssim_rgb', 'temporal_error', 'edge_mae']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include', required=True, help='self-motion,published or either completed stage')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    stages = args.include.split(',')
    assert stages and len(set(stages)) == len(stages) and set(stages) <= {'self-motion', 'published'}
    reference = BASE/'results/exploration-r30-eval-references/per_sequence.jsonl'
    refs = [json.loads(line) for line in reference.read_text().splitlines()]
    refs = [r for r in refs if r['mode'] == 'copy' and r['objective'] == 'mse' and not r['input_reset']]
    rows = [{**r, 'method': 'raw' if r['fusion'] == 'raw' else 'shared_motion_post05'} for r in refs]
    assert len(rows) == 24
    old_raw = {(r['scene'], r['train_seed']): r for r in rows if r['method'] == 'raw'}
    sources = {str(reference.relative_to(ROOT)): digest(reference)}
    overlap = None
    if 'published' in stages:
        path = BASE/'data/external-davis/fastdvdnet-training-overlap.json'
        overlap = json.loads(path.read_text())
        assert set(overlap['overlap_R30_DEV']) == {'bear', 'lucia', 'walking'}
        sources[str(path.relative_to(ROOT))] = digest(path)
    for stage in stages:
        folder = BASE/'results'/('exploration-r30-self-motion' if stage == 'self-motion' else 'exploration-r35-published')
        runtime = json.loads((folder/'runtime.json').read_text())
        assert runtime['completed_normally'] and not runtime['cuda_initialized']
        assert runtime['rows'] == (24 if stage == 'self-motion' else 36)
        values = [json.loads(line) for line in (folder/'per_sequence.jsonl').read_text().splitlines()]
        assert len(values) == runtime['rows']
        for row in values:
            if row['mode'] == 'raw':
                old = old_raw[(row['scene'], row['train_seed'])]
                assert max(abs(row[m]-old[m]) for m in METRICS) <= 1e-6
            else:
                method = row['mode'] if stage == 'self-motion' else 'fastdvd_sigma15_'+row['mode']
                rows.append({**row, 'method': method})
        for name in ['runtime.json', 'per_sequence.jsonl', 'config.json']:
            path = folder/name
            sources[str(path.relative_to(ROOT))] = digest(path)
    lookup = {(r['method'], r['scene'], r['train_seed']): r for r in rows}
    assert len(lookup) == len(rows)
    methods = sorted({r['method'] for r in rows})
    groups = defaultdict(list)
    for row in rows:
        groups[(row['method'], row['train_seed'])].append(row)
    seeds, domains, paired = [], [], []
    for (method, seed), values in sorted(groups.items()):
        assert len(values) == 6 and len({r['source_video'] for r in values}) == 3
        seeds.append({'method': method, 'seed': seed, **{m: mean(r[m] for r in values) for m in METRICS}})
    for method in methods:
        values = [r for r in seeds if r['method'] == method]
        assert {r['seed'] for r in values} == {0, 1}
        domains.append({'method': method, 'source_videos': 3, 'noise_conditions_per_video': 2, 'seeds': 2,
                        **{m: mean(r[m] for r in values) for m in METRICS}})
    for row in rows:
        if row['method'] in ['raw', 'shared_motion_post05']:
            continue
        comparisons = ['raw', 'shared_motion_post05']
        if row['method'] == 'fastdvd_sigma15_window':
            comparisons.append('fastdvd_sigma15_five_current')
        for reference_method in comparisons:
            other = lookup[(reference_method, row['scene'], row['train_seed'])]
            paired.append({'method': row['method'], 'reference': reference_method, 'scene': row['scene'],
                           'source_video': row['source_video'], 'seed': row['train_seed'],
                           **{'delta_'+m: row[m]-other[m] for m in METRICS},
                           'temporal_ratio': row['temporal_error']/other['temporal_error']})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    for filename, value in [('domain_seed_summary.json', seeds), ('domain_summary.json', domains),
                            ('paired_by_condition.json', paired), ('audit.json', {'passed': True, 'included_stages': stages,
                             'methods': methods, 'rows': len(rows), 'source_sha256': sources,
                             'published_documented_training_overlap_DEV': overlap['overlap_R30_DEV'] if overlap else None,
                             'scope': 'Same R26 plain MSE base weights and common CPU R30 input; three DEV source videos, not six independent benchmarks or a phone test.'})]:
        (out/filename).write_text(json.dumps(value, indent=2)+'\n')
    lines = ['# Same-base external development baselines', '',
             'R26 plain MSE, both seeds, common CPU. Equal means over three source videos × two noise conditions × two training seeds. Custom synthetic lowlight DEV; no official benchmark or phone measurement.', '',
             '|Method|PSNR|SSIM|Temporal↓|Edge↓|', '|---|---:|---:|---:|---:|']
    for row in domains:
        lines.append('|'+row['method']+'|'+'|'.join(f'{row[m]:.6f}' for m in METRICS)+'|')
    if overlap:
        lines += ['', '**Prior-training overlap:** the official FastDVDnet training list includes all three R30 DEV videos, all 60 DAVIS TRAIN IDs and all 30 VAL IDs. These rows are development/transfer evidence, never held-out-video generalization. The list is not a per-weight frame-sampling trace. See the versioned training-overlap metadata.']
    lines += ['', 'Self-motion post reuses its own enhancement and needs no extra teacher CNN. Shared-motion post uses the original fixed teacher flow for quality control. FastDVDnet window waits two future frames; five_current removes real temporal input with identical denoiser capacity. CPU stage timings exclude phone/camera behavior and cannot be reported as mobile latency.', '',
              '|Method|Seed|PSNR|SSIM|Temporal↓|Edge↓|', '|---|---:|---:|---:|---:|---:|']
    for row in seeds:
        lines.append('|'+row['method']+'|'+str(row['seed'])+'|'+'|'.join(f'{row[m]:.6f}' for m in METRICS)+'|')
    (out/'tables.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'included_stages': stages, 'rows': len(rows), 'methods': len(methods)}))


if __name__ == '__main__':
    main()
