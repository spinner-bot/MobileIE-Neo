"""Complete 140-condition joint table and predeclared complement-versus-control gates."""
from collections import defaultdict
import json

import numpy as np

from shared_geometry_io_r88 import ROOT, BASE, PROCESS, verify, lines, digest, write


def key(row):
    return row['seed'], row['scene'], row['label']


def main():
    verify(json.loads((PROCESS/'evaluation-source-freeze001.json').read_text()))
    reference = BASE/'report/exploration-r86-joint/all_conditions.jsonl'
    assert json.loads((reference.parent/'audit.json').read_text())['completed']
    rows = lines(reference)
    assert len(rows) == 60 and {r['label'] for r in rows} == {
        'MobileIE_raw', 'MobileIE_recursive_self05', 'DIS_recursive_self05'}
    sources = {str(reference.relative_to(ROOT)): digest(reference)}
    new = []
    for seed in (0, 1):
        root = BASE/f'results/exploration-r88-eval-seed{seed}'
        runtime = json.loads((root/'runtime.json').read_text())
        assert runtime['completed_normally'] and runtime['quality_rows'] == 40 and runtime['frame_rows'] == 1792
        path = root/'per_sequence.jsonl'; values = lines(path)
        assert len(values) == 40
        new += values; sources[str(path.relative_to(ROOT))] = digest(path)
    root = BASE/'results/exploration-r88-perceptual001'
    runtime = json.loads((root/'runtime.json').read_text())
    assert runtime['completed_normally'] and runtime['quality_rows'] == 80 and runtime['comparison_frames'] == 3584
    path = root/'per_sequence.jsonl'; perceptual = lines(path)
    lookup = {key(r): r for r in perceptual}
    assert len(perceptual) == len(lookup) == 80 and set(lookup) == {key(r) for r in new}
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
    detail = {key(r): r for r in rows}; comparisons = []
    for domain in ('davis_train_pilot', 'paired_real_ND_DEV'):
        for seed in (0, 1):
            for mode in ('plain', 'complement'):
                label = mode+'_true'; candidate = index[domain, seed, label]
                controls = ['MobileIE_raw', 'MobileIE_recursive_self05', 'DIS_recursive_self05', mode+'_zero']
                if mode == 'complement':
                    controls.append('plain_true')
                for control in controls:
                    ref = index[domain, seed, control]
                    psnr = candidate['psnr_rgb']-ref['psnr_rgb']
                    jod = candidate['JOD']-ref['JOD']
                    ratio = candidate['temporal_error']/ref['temporal_error']
                    deltas = [dict(scene=r['scene'], PSNR_delta=r['psnr_rgb']-detail[seed, r['scene'], control]['psnr_rgb'],
                        JOD_delta=r['JOD']-detail[seed, r['scene'], control]['JOD'])
                        for r in rows if r['domain'] == domain and r['seed'] == seed and r['label'] == label]
                    worst_psnr, worst_jod = min(r['PSNR_delta'] for r in deltas), min(r['JOD_delta'] for r in deltas)
                    quality = psnr >= -.05 and jod >= -.02 and worst_psnr >= -.30 and worst_jod >= -.10
                    classic = control in ('MobileIE_recursive_self05', 'DIS_recursive_self05')
                    cap_a = .90 if classic else (None if control == 'MobileIE_raw' else .95)
                    cap_b = 1.05 if classic else (None if control == 'MobileIE_raw' else .95)
                    comparisons.append(dict(domain=domain, seed=seed, candidate=label, reference=control,
                        PSNR_delta=psnr, JOD_delta=jod, Etemp_ratio=ratio,
                        worst_condition_PSNR_delta=worst_psnr, worst_condition_JOD_delta=worst_jod,
                        condition_deltas=deltas, Etemp_max_A=cap_a, Etemp_max_B=cap_b,
                        passed_quality_protection=quality,
                        passed_A_quality=quality and (cap_a is None or ratio <= cap_a),
                        passed_B_quality=quality and (cap_b is None or ratio <= cap_b),
                        primary_candidate_gate=mode == 'complement'))
    assert len(comparisons) == 36
    primary = [r for r in comparisons if r['primary_candidate_gate']]
    assert len(primary) == 20
    a, b = all(r['passed_A_quality'] for r in primary), all(r['passed_B_quality'] for r in primary)
    out = BASE/'report/exploration-r88-joint'; out.mkdir(exist_ok=False)
    (out/'all_conditions.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows))
    write(out/'domain_seed_means.json', means); write(out/'comparisons.json', comparisons); write(out/'sources.json', sources)
    table = ['# R88 完整共同 DEV 表', '',
        '4507部署参数；plain学432、complement学384（rank12时），冻结F0。不是容量/优化等价或零推理开销。', '',
        '|域|seed|控制|PSNR|Etemp|JOD|', '|---|---:|---|---:|---:|---:|']
    for row in means:
        table.append(f"|{row['domain']}|{row['seed']}|{row['label']}|{row['psnr_rgb']:.6f}|{row['temporal_error']:.6f}|{row['JOD']:.6f}|")
    (out/'tables.md').write_text('\n'.join(table)+'\n')
    write(out/'audit.json', dict(completed=True, new_quality_rows=80, reused_reference_rows=60,
        all_quality_rows=140, domain_seed_means=28, comparisons=36, primary_comparisons=20,
        A_quality_eligible=a, B_quality_eligible=b, cost_measurement_eligible=a or b,
        A_cost_median_ratio_max=1.05, B_cost_median_ratio_max=.80,
        full_deployment_cost_measured=False, formal_optimizer_updates=8000,
        decision='Complete cost qualification pending' if a or b else 'STOP this descriptor/solver local grid',
        new_method_established=False, novelty_established=False, natural_night_claim=False,
        independent_final=False, phone_measured=False))


if __name__ == '__main__':
    main()
