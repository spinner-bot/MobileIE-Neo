"""Describe every completed CPU pipeline run; no mobile-speed inference."""
from collections import defaultdict
import json
from statistics import mean, median

from dynamic_training_r34 import BASE, ROOT, digest


def main():
    folder = BASE/'results/exploration-r34-runtime'
    runtime = json.loads((folder/'runtime.json').read_text())
    config = json.loads((folder/'config.json').read_text())
    assert runtime['completed_normally'] and runtime['runs'] == 42 and not runtime['cuda_initialized']
    rows = [json.loads(line) for line in (folder/'per_run.jsonl').read_text().splitlines()]
    assert len(rows) == 42
    groups = defaultdict(list)
    for row in rows:
        assert row['frames'] == 16 and row['output_shape'] == [16, 480, 854, 3]
        assert row['finite_before_and_after_clipping']
        groups[(row['model'], row['history'], row['fusion'])].append(row)
    assert len(groups) == 7
    summaries, seed_rows = [], []
    for (model, history, fusion), values in sorted(groups.items()):
        assert len(values) == 6 and {(r['seed'], r['repetition']) for r in values} == {(s, r) for s in [0, 1] for r in range(3)}
        storage = {r['online_history_storage_bytes'] for r in values}
        assert len(storage) == 1
        if history == 'causal':
            assert storage == {8*120*214*4}
        elif fusion == 'self_motion_post05':
            assert storage == {3*480*854*4}
        else:
            assert storage == {0}
        times = [1000*r['wall_seconds']/r['frames'] for r in values]
        summaries.append({'model': model, 'history': history, 'fusion': fusion,
                          'parameters': config['parameters'][f'0/{model}'], 'median_pipeline_ms_per_frame': median(times),
                          'mean_pipeline_ms_per_frame': mean(times), 'min_pipeline_ms_per_frame': min(times),
                          'max_pipeline_ms_per_frame': max(times), 'online_history_storage_bytes': next(iter(storage)),
                          'forward_layout_ms': 1000*mean(t['forward_layout_seconds'] for r in values for t in r['per_frame']),
                          'motion_ms': 1000*mean(t['motion_seconds'] for r in values for t in r['per_frame']),
                          'warp_blend_ms': 1000*mean(t['warp_blend_seconds'] for r in values for t in r['per_frame'])})
        for seed in [0, 1]:
            selected = [1000*r['wall_seconds']/r['frames'] for r in values if r['seed'] == seed]
            seed_rows.append({'model': model, 'history': history, 'fusion': fusion, 'seed': seed,
                              'median_pipeline_ms_per_frame': median(selected), 'all_repetitions_ms': selected})
    out = BASE/'report/exploration-r34-runtime'
    out.mkdir(parents=True, exist_ok=False)
    audit = {'passed': True, 'runs': 42, 'mappings': 7, 'frames_per_run': 16, 'repetitions_per_seed': 3,
             'source_sha256': {str((folder/name).relative_to(ROOT)): digest(folder/name) for name in
                              ['runtime.json', 'config.json', 'per_run.jsonl', 'execution_order.json']},
             'scope': 'Median of six run-average frame costs per mapping; fixed single DEV workload, shared-server CPU, not phone, energy, peak memory or quality evidence.'}
    for name, value in [('cost.json', summaries), ('per_seed.json', seed_rows), ('audit.json', audit)]:
        (out/name).write_text(json.dumps(value, indent=2)+'\n')
    lines = ['# R34 final-model CPU pipeline cost', '',
             'Resident bear_photons20 frames0–15, 480×854, FP32, two Torch/one OpenCV threads. Three rotated/reversed rounds per training seed. Each value below is the median of six run-average frame costs; ranges show all runs. No phone, energy, peak-memory or quality claim.', '',
             '|Model|History|Fusion|Parameters|Median ms/frame|Min–max|Online history MiB|',
             '|---|---|---|---:|---:|---:|---:|']
    for r in summaries:
        lines.append(f"|{r['model']}|{r['history']}|{r['fusion']}|{r['parameters']}|{r['median_pipeline_ms_per_frame']:.3f}|{r['min_pipeline_ms_per_frame']:.3f}–{r['max_pipeline_ms_per_frame']:.3f}|{r['online_history_storage_bytes']/2**20:.3f}|")
    lines += ['', 'Self-motion post uses its own raw enhancement, no extra teacher. Actual causal uses its low-resolution state, not external flow. Timing includes tensor preparation, prediction, layout/clipping and required motion/warp/blend; loop output references are retained. Final finite checks, stack, IO and model loading are outside timing.']
    (out/'tables.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'runs': 42, 'mappings': 7}))


if __name__ == '__main__':
    main()
