"""Full R74 quality/perception table and16 preregistered gates, no prefix ranking."""
import argparse
from collections import defaultdict
import json
import numpy as np

from motion_transport_r74 import BASE, ROOT
from check_motion_transport_r74 import write
from dynamic_training_r34 import digest
from evaluate_motion_transport_r74 import reference_quality, key
from evaluate_cvvdp_references_r66 import lines


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--perceptual', action='store_true'); args = parser.parse_args()
    rows, sources = reference_quality(); new = []
    for seed in [0, 1]:
        folder = BASE/f'results/exploration-r74-eval-seed{seed}'
        terminal = json.loads((folder/'runtime.json').read_text())
        assert terminal['completed_normally'] and (terminal['quality_rows'], terminal['frame_rows']) == (40, 1792)
        assert (terminal['motion_sequences'], terminal['nonfirst_motion_pairs']) == (20, 876)
        path = folder/'per_sequence.jsonl'; selected = lines(path); assert len(selected) == 40
        sources[str(path.relative_to(ROOT))] = digest(path); new += selected
    assert len(new) == 80
    if args.perceptual:
        folder = BASE/'results/exploration-r74-perceptual001'
        terminal = json.loads((folder/'runtime.json').read_text())
        assert terminal['completed_normally'] and (terminal['quality_rows'], terminal['comparison_frames']) == (80, 3584)
        path = folder/'per_sequence.jsonl'; perceptual = lines(path); lookup = {key(r): r for r in perceptual}
        assert len(perceptual) == len(lookup) == 80 and set(lookup) == {key(r) for r in new}
        sources[str(path.relative_to(ROOT))] = digest(path)
        new = [dict(r, JOD=lookup[key(r)]['JOD']) for r in new]
    rows += new; assert len(rows) == len({key(r) for r in rows}) == 120
    groups = defaultdict(list)
    for r in rows: groups[r['domain'], r['seed'], r['model'], r['history'], r['method']].append(r)
    assert len(groups) == 24
    fields = ['psnr_rgb', 'ssim_rgb', 'temporal_error', 'edge_mae']+(['JOD'] if args.perceptual else [])
    means = []
    for (domain, seed, model, history, method), entries in groups.items():
        assert len(entries) == (6 if domain == 'davis_train_pilot' else 4)
        means.append(dict(domain=domain, seed=seed, model=model, history=history, method=method,
              parameters=entries[0]['parameters'], spatial_updates=8000,
              motion_updates=0 if model == 'R44_plain' else 4000, conditions=len(entries),
              **{field: float(np.mean([r[field] for r in entries])) for field in fields}))
    gates = []
    if args.perceptual:
        index = {(r['domain'], r['seed'], r['model'], r['history'], r['method']): r for r in means}
        detail = {key(r): r for r in rows}
        for domain in ['davis_train_pilot', 'paired_real_ND_DEV']:
            for seed in [0, 1]:
                main = index[domain, seed, 'proxy', 'causal', 'raw']
                for model, history, method, threshold in [('R44_plain', 'current', 'raw', .95),
                      ('proxy', 'zero_flow', 'raw', .95), ('task', 'causal', 'raw', .95),
                      ('R44_plain', 'current', 'recursive_self05', 1.02)]:
                    reference = index[domain, seed, model, history, method]
                    worst = min(r['psnr_rgb']-detail[seed, r['scene'], model, history, method]['psnr_rgb']
                                for r in rows if (r['domain'], r['seed'], r['model'], r['history'], r['method']) ==
                                (domain, seed, 'proxy', 'causal', 'raw'))
                    psnr = main['psnr_rgb']-reference['psnr_rgb']; jod = main['JOD']-reference['JOD']
                    ratio = main['temporal_error']/reference['temporal_error']
                    gates.append(dict(domain=domain, seed=seed, reference=[model, history, method],
                          PSNR_delta=psnr, JOD_delta=jod, Etemp_ratio=ratio, Etemp_max=threshold,
                          worst_condition_PSNR_delta=worst,
                          passed=psnr >= -.05 and jod >= -.02 and ratio <= threshold and worst >= -.30))
        assert len(gates) == 16
    out = BASE/('report/exploration-r74-joint' if args.perceptual else 'report/exploration-r74-quality')
    out.mkdir(exist_ok=False)
    (out/'all_conditions.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows))
    write(out/'domain_seed_means.json', means); write(out/'gates.json', gates); write(out/'sources.json', sources)
    table = ['# R74完整固定对照', '', '6 SYN条件来自3视频，4 ND开发源；无独立最终或手机结果。空间固定8000，两个motion各新4000，不合计为空间12000。', '',
             '|域|seed|模型|history|输出|总参数|PSNR|Etemp|JOD|', '|---|---:|---|---|---|---:|---:|---:|---:|']
    for r in means:
        jod = f"{r['JOD']:.6f}" if args.perceptual else '—'
        table.append(f"|{r['domain']}|{r['seed']}|{r['model']}|{r['history']}|{r['method']}|{r['parameters']}|{r['psnr_rgb']:.6f}|{r['temporal_error']:.6f}|{jod}|")
    (out/'tables.md').write_text('\n'.join(table)+'\n')
    write(out/'audit.json', dict(completed=True, new_quality_rows=80, all_quality_rows=120, domain_seed_means=24,
          perceptual_included=args.perceptual, all_gates_pass=None if not args.perceptual else all(r['passed'] for r in gates),
          formal_motion_updates=16000, new_spatial_updates=0, new_enhancement_CNN=0,
          motion_sequences=40, nonfirst_motion_pairs=1752, fair_cost_measured=False,
          phone_measured=False, independent_final=False, old_gates_rewritten=False))


if __name__ == '__main__': main()
