"""All completed native REDS reference conditions, with optional complete perception."""
import argparse
from collections import defaultdict
import json
import numpy as np

from reds_references_r72 import BASE, DATA, digest, lines, stamp, write


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--include-perceptual', action='store_true')
    args = parser.parse_args(); sources = {}

    def read(path):
        sources[str(path)] = digest(path)
        return json.loads(path.read_text())

    def read_lines(path):
        sources[str(path)] = digest(path)
        return lines(path)

    assert read(DATA / 'runtime.json')['completed_normally']
    rows = read_lines(DATA / 'input_per_sequence.jsonl'); assert len(rows) == 8
    for seed in [0, 1]:
        folder = BASE / f'results/exploration-r72-quality-seed{seed}'
        terminal = read(folder / 'runtime.json')
        assert terminal['completed_normally'] and (terminal['quality_rows'], terminal['frame_rows']) == (72, 4608)
        rows += read_lines(folder / 'per_sequence.jsonl')
    assert len(rows) == 152
    key = lambda r: (r['scene'], r['family'], r['seed'], r['method'])
    assert len({key(r) for r in rows}) == 152
    if args.include_perceptual:
        folder = BASE / 'results/exploration-r72-perceptual001'
        terminal = read(folder / 'runtime.json')
        assert terminal['completed_normally'] and (terminal['perceptual_rows'], terminal['comparison_frames']) == (156, 9984)
        metric = read_lines(folder / 'per_sequence.jsonl')
        gt = [r for r in metric if r['family'] == 'GT']; assert len(gt) == 4
        assert all(abs(r['JOD'] - 10) <= 1e-4 for r in gt)
        lookup = {key(r): r for r in metric if r['family'] != 'GT'}; assert len(lookup) == 152
        assert set(lookup) == {key(r) for r in rows}
        rows = [dict(r, JOD=lookup[key(r)]['JOD']) for r in rows]
    groups = defaultdict(list)
    for r in rows:
        groups[r['family'], r['seed'], r['method'], r['photons']].append(r)
    assert len(groups) == 38
    means = []
    fields = ['psnr_rgb', 'ssim_rgb', 'temporal_error', 'edge_mae'] + (['JOD'] if args.include_perceptual else [])
    for (family, seed, method, photons), entries in groups.items():
        assert len(entries) == 4 and len({r['source_video'] for r in entries}) == 4
        assert all(r[field] is not None and np.isfinite(r[field]) for r in entries for field in fields)
        means.append(dict(family=family, seed=seed, method=method, photons=photons, source_videos=4,
                          conditions=4, parameters=entries[0]['parameters'],
                          **{field: float(np.mean([r[field] for r in entries])) for field in fields}))
    index = {(r['family'], r['seed'], r['method'], r['photons']): r for r in means}
    effects = []
    for seed in [0, 1]:
        for photons in [20, 80]:
            for method in ['raw', 'recursive_self05']:
                l1 = index['MobileIE_L1', seed, method, photons]
                mse = index['MobileIE_MSE', seed, method, photons]
                effect = dict(seed=seed, photons=photons, method=method,
                              PSNR_delta=l1['psnr_rgb'] - mse['psnr_rgb'],
                              Etemp_ratio=l1['temporal_error'] / mse['temporal_error'])
                if args.include_perceptual:
                    effect['JOD_delta'] = l1['JOD'] - mse['JOD']
                effects.append(effect)
    out = BASE / ('report/exploration-r72-reds-joint' if args.include_perceptual else 'report/exploration-r72-reds-quality')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'all_conditions.jsonl').write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in rows))
    write(out / 'condition_means.json', means); write(out / 'objective_effects.json', effects)
    write(out / 'sources.json', sources)
    table = ['# 固定REDS来源开发参考', '', '4原视频/8噪声条件，512唯一低光观察；原生720p/来源24fps。非自然夜间、官方LLVE或独立最终测试；不据此选模型或宣称新方法。', '',
             '|模型|seed|输出|光子|参数|PSNR|SSIM|Etemp|edge|JOD|', '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in means:
        values = [f'{r[k]:.6f}' for k in ['psnr_rgb', 'ssim_rgb', 'temporal_error', 'edge_mae']]
        values.append(f"{r['JOD']:.6f}" if args.include_perceptual else '—')
        table.append(f"|{r['family']}|{r['seed']}|{r['method']}|{r['photons']}|{r['parameters']}|" + '|'.join(values) + '|')
    (out / 'tables.md').write_text('\n'.join(table) + '\n')
    audit = dict(completed=True, at=stamp(), quality_rows=152, frame_rows=9728,
                 learned_output_rows=144, CNN_sequences=80, CNN_centers=5120, SGD=0,
                 unique_observations=512, independent_source_videos=4, condition_means=38,
                 perceptual_included=args.include_perceptual, source_fps=24,
                 new_method=False, phone=False, independent_final=False, old_gates_rewritten=False)
    write(out / 'audit.json', audit); print(json.dumps(audit))


if __name__ == '__main__':
    main()
