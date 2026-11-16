"""Complete joint evidence and all16 registered continuation comparisons."""
from collections import defaultdict
import json

import numpy as np

from noise_role_io_r79 import ROOT, BASE, digest, write
from noise_role_evidence_r79 import verify_evaluation
from causal_reference_evaluation_r75 import reference_quality
from causal_reference_io_r75 import lines


def key(row):
    return row['seed'], row['scene'], row['label']


def main():
    verify_evaluation()
    rows, sources = reference_quality()
    new = []
    for seed in (0, 1):
        run = BASE/f'results/exploration-r79-eval-seed{seed}'
        terminal = json.loads((run/'runtime.json').read_text())
        assert terminal['completed_normally'] and terminal['quality_rows'] == 50 and terminal['frame_rows'] == 2240
        path = run/'per_sequence.jsonl'; values = lines(path)
        assert len(values) == 50
        new += values; sources[str(path.relative_to(ROOT))] = digest(path)
    run = BASE/'results/exploration-r79-perceptual001'
    terminal = json.loads((run/'runtime.json').read_text())
    assert terminal['completed_normally'] and terminal['quality_rows'] == 100 and terminal['comparison_frames'] == 4480
    path = run/'per_sequence.jsonl'; perceptual = lines(path)
    lookup = {key(r): r for r in perceptual}
    assert len(perceptual) == len(lookup) == 100 and set(lookup) == {key(r) for r in new}
    sources[str(path.relative_to(ROOT))] = digest(path)
    rows += [dict(row, JOD=lookup[key(row)]['JOD']) for row in new]
    assert len(rows) == len({key(r) for r in rows}) == 140
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['domain'], row['seed'], row['label']].append(row)
    means = []
    for (domain, seed, label), values in grouped.items():
        assert len(values) == (6 if domain == 'davis_train_pilot' else 4)
        means.append(dict(domain=domain, seed=seed, label=label, conditions=len(values),
            **{field: float(np.mean([r[field] for r in values])) for field in
               ('psnr_rgb', 'ssim_rgb', 'temporal_error', 'edge_mae', 'JOD')}))
    assert len(means) == 28
    index = {(r['domain'], r['seed'], r['label']): r for r in means}
    gates = []
    detail = {key(r): r for r in rows}
    for domain in ('davis_train_pilot', 'paired_real_ND_DEV'):
        for seed in (0, 1):
            candidate = index[domain, seed, 'curriculum_causal']
            for label, threshold in (('balanced_causal', .95), ('spatial_current', .95),
                                     ('curriculum_current', .95), ('MobileIE_recursive_self05', .90)):
                reference = index[domain, seed, label]
                psnr = candidate['psnr_rgb']-reference['psnr_rgb']
                jod = candidate['JOD']-reference['JOD']
                ratio = candidate['temporal_error']/reference['temporal_error']
                deltas = [dict(scene=r['scene'], PSNR_delta=r['psnr_rgb']-detail[seed, r['scene'], label]['psnr_rgb'],
                               JOD_delta=r['JOD']-detail[seed, r['scene'], label]['JOD'])
                          for r in rows if r['domain'] == domain and r['seed'] == seed and r['label'] == 'curriculum_causal']
                gates.append(dict(domain=domain, seed=seed, reference=label, PSNR_delta=psnr, JOD_delta=jod,
                    Etemp_ratio=ratio, Etemp_max=threshold, condition_deltas=deltas,
                    passed=psnr >= -.05 and jod >= -.02 and ratio <= threshold))
    assert len(gates) == 16
    out = BASE/'report/exploration-r79-joint'; out.mkdir(exist_ok=False)
    (out/'all_conditions.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows))
    write(out/'domain_seed_means.json', means); write(out/'gates.json', gates); write(out/'sources.json', sources)
    table = ['# R79完整训练信息条件对照', '',
        '重复DEV：六SYN条件来自三源、四ND源。所有新控制均24955参数/4000更新；既有MobileIE参考4075参数/8000更新，非公平架构或训练算力排名。', '',
        '|域|seed|控制|PSNR|Etemp|JOD|', '|---|---:|---|---:|---:|---:|']
    for r in means:
        table.append(f"|{r['domain']}|{r['seed']}|{r['label']}|{r['psnr_rgb']:.6f}|{r['temporal_error']:.6f}|{r['JOD']:.6f}|")
    (out/'tables.md').write_text('\n'.join(table)+'\n')
    passed = all(r['passed'] for r in gates)
    write(out/'audit.json', dict(completed=True, new_quality_rows=100, reused_reference_rows=40,
        all_quality_rows=140, domain_seed_means=28, comparisons=16, all_gates_pass=passed,
        formal_optimizer_updates=24000, decision='further distinct method design allowed' if passed else 'STOP local curriculum grid',
        new_method_established=False, natural_night_claim=False, independent_final=False, phone_measured=False))


if __name__ == '__main__':
    main()
