"""Full R73 controls and fixed old references; no prefix ranking or selection."""
import argparse
from collections import defaultdict
from datetime import datetime
import json
import time
import numpy as np

from native_moment_core_r73 import BASE, ROOT, PROTOCOL
from check_native_moment_r73 import write
from evaluate_native_moment_r73 import append
from dynamic_training_r34 import digest
from render_dense_history_r64 import recursive


def lines(path): return [json.loads(line) for line in path.read_text().splitlines()]
def key(row): return row['seed'], row['scene'], row['model'], row['history'], row['method']


def old_quality():
    path = BASE/'report/exploration-r70-classical-joint/all_conditions.jsonl'
    assert json.loads((path.parent/'audit.json').read_text())['completed']
    rows = []
    for r in lines(path):
        if r['label'] in ['MobileIE_raw', 'MobileIE_recursive_self05']:
            rows.append(dict(r, model='R44_plain', history='current', parameters=4075))
    assert len(rows) == 40
    sources = {str(path.relative_to(ROOT)): digest(path)}
    for seed in [0, 1]:
        folder = BASE/f'results/exploration-r64-stage001-eval-seed{seed}'
        assert json.loads((folder/'runtime.json').read_text())['completed_normally']
        path = folder/'per_sequence.jsonl'; sources[str(path.relative_to(ROOT))] = digest(path)
        selected = [r for r in lines(path) if r['model'] == 'plain']; assert len(selected) == 20
        rows += [dict(r, seed=seed, model='R64_plain', project_updates=12000) for r in selected]
    assert len(rows) == 80
    return rows, sources


def prepare_old_predictions():
    out = BASE/'process/exploration-r73/references001'; out.mkdir(exist_ok=False)
    cache = ROOT/'.cache/experiments/exploration-r73/references001'; cache.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter(); complete = False; count = 0
    try:
        write(out/'config.json', dict(protocol_sha256=digest(PROTOCOL),
              source_sha256={n: digest(BASE/'scripts'/n) for n in
                  ['render_dense_history_r64.py', 'half_motion_r4.py', 'aligned_raw_history_r11.py', 'aggregation_controls_r20.py']},
              scope='Reconstruct original R64 plain own-recursion before perceptual GPU. No CNN/SGD or recomputed old quality.'))
        for seed in [0, 1]:
            path = BASE/f'results/exploration-r64-stage001-eval-seed{seed}/prediction_manifest.jsonl'
            selected = [r for r in lines(path) if r['model'] == 'plain']; assert len(selected) == 10
            for entry in selected:
                raw_path = ROOT/entry['path']; assert digest(raw_path) == entry['sha256']
                raw = np.load(raw_path, mmap_mode='r', allow_pickle=False)
                rec = recursive(raw)
                for method, value in [('raw', raw), ('recursive_self05', rec)]:
                    output = raw_path if method == 'raw' else cache/f'seed{seed}--{entry["scene"]}--recursive.npy'
                    if method != 'raw': np.save(output, value, allow_pickle=False)
                    append(out/'prediction_manifest.jsonl', dict(entry, model='R64_plain', seed=seed,
                           history='current', method=method, path=str(output.relative_to(ROOT)), sha256=digest(output),
                           shape=list(value.shape), source_manifest=str(path.relative_to(ROOT)), source_manifest_sha256=digest(path)))
                    count += 1
        assert count == 40; complete = True
    finally:
        write(out/'runtime.json', dict(completed_normally=complete, rows=count, CNN=0, SGD=0, GPU=0,
              seconds=time.perf_counter()-start, ended=datetime.now().astimezone().isoformat()))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--perceptual', action='store_true'); args = parser.parse_args()
    rows, sources = old_quality(); new = []
    for seed in [0, 1]:
        path = BASE/f'results/exploration-r73-eval-seed{seed}'
        terminal = json.loads((path/'runtime.json').read_text())
        assert terminal['completed_normally'] and (terminal['quality_rows'], terminal['frame_rows']) == (100, 4480)
        file = path/'per_sequence.jsonl'; sources[str(file.relative_to(ROOT))] = digest(file)
        selected = lines(file); assert len(selected) == 100; new += selected
    rows += new; assert len(rows) == 280 and len({key(r) for r in rows}) == 280
    if not args.perceptual:
        prepare_old_predictions()
    else:
        folder = BASE/'results/exploration-r73-perceptual001'
        terminal = json.loads((folder/'runtime.json').read_text())
        assert terminal['completed_normally'] and (terminal['quality_rows'], terminal['comparison_frames']) == (240, 10752)
        path = folder/'per_sequence.jsonl'; sources[str(path.relative_to(ROOT))] = digest(path)
        lookup = {key(r): r for r in lines(path)}; assert len(lookup) == 240
        assert set(lookup) == {key(r) for r in rows if r['model'] != 'R44_plain'}
        rows = [r if r['model'] == 'R44_plain' else dict(r, JOD=lookup[key(r)]['JOD']) for r in rows]
    groups = defaultdict(list)
    for r in rows: groups[r['domain'], r['seed'], r['model'], r['history'], r['method']].append(r)
    assert len(groups) == 56
    fields = ['psnr_rgb', 'ssim_rgb', 'temporal_error', 'edge_mae'] + (['JOD'] if args.perceptual else [])
    means = []
    for (domain, seed, model, history, method), entries in groups.items():
        assert len(entries) == (6 if domain == 'davis_train_pilot' else 4)
        means.append(dict(domain=domain, seed=seed, model=model, history=history, method=method,
                          parameters=entries[0]['parameters'], conditions=len(entries),
                          **{field: float(np.mean([r[field] for r in entries])) for field in fields}))
    gates = []
    if args.perceptual:
        index = {(r['domain'], r['seed'], r['model'], r['history'], r['method']): r for r in means}
        details = {key(r): r for r in rows}
        for domain in ['davis_train_pilot', 'paired_real_ND_DEV']:
            for seed in [0, 1]:
                main = index[domain, seed, 'moment', 'causal', 'raw']
                for model, history, method, threshold in [('spatial', 'current', 'raw', .95), ('mean', 'causal', 'raw', .95),
                      ('moment', 'current', 'raw', .95), ('R44_plain', 'current', 'recursive_self05', .90),
                      ('R64_plain', 'current', 'recursive_self05', .90)]:
                    reference = index[domain, seed, model, history, method]
                    worst = min(r['psnr_rgb']-details[seed, r['scene'], model, history, method]['psnr_rgb']
                                for r in rows if r['domain'] == domain and r['seed'] == seed and
                                (r['model'], r['history'], r['method']) == ('moment', 'causal', 'raw'))
                    psnr = main['psnr_rgb']-reference['psnr_rgb']; jod = main['JOD']-reference['JOD']
                    ratio = main['temporal_error']/reference['temporal_error']
                    gates.append(dict(domain=domain, seed=seed, reference=[model, history, method],
                          PSNR_delta=psnr, JOD_delta=jod, Etemp_ratio=ratio, Etemp_max=threshold,
                          worst_condition_PSNR_delta=worst,
                          passed=psnr >= -.05 and jod >= -.02 and ratio <= threshold and worst >= -.30))
        assert len(gates) == 20
    out = BASE/('report/exploration-r73-joint' if args.perceptual else 'report/exploration-r73-quality')
    out.mkdir(exist_ok=False)
    (out/'all_conditions.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows))
    write(out/'domain_seed_means.json', means); write(out/'gates.json', gates); write(out/'sources.json', sources)
    table = ['# R73完整固定控制', '', '6 SYN条件来自3原视频，4 ND开发源；不是独立最终或手机结果。', '',
             '|域|seed|模型|history|输出|参数|PSNR|Etemp|JOD|', '|---|---:|---|---|---|---:|---:|---:|---:|']
    for r in means:
        jod = f"{r['JOD']:.6f}" if args.perceptual else '—'
        table.append(f"|{r['domain']}|{r['seed']}|{r['model']}|{r['history']}|{r['method']}|{r['parameters']}|{r['psnr_rgb']:.6f}|{r['temporal_error']:.6f}|{jod}|")
    (out/'tables.md').write_text('\n'.join(table)+'\n')
    write(out/'audit.json', dict(completed=True, new_quality_rows=200, all_quality_rows=280, domain_seed_means=56,
          perceptual_included=args.perceptual, all_gates_pass=None if not args.perceptual else all(r['passed'] for r in gates),
          new_CNN_sequences=100, new_CNN_centers=4480, formal_SGD_updates=24000,
          phone_measured=False, independent_final=False, old_gates_rewritten=False))


if __name__ == '__main__': main()
