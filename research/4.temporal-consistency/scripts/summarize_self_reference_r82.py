"""All140 conditions and registered 24 comparisons; no interim ranking or cost claims."""
from collections import defaultdict
import json

import numpy as np

from self_reference_io_r82 import ROOT, BASE, digest, write, lines, verify
from causal_reference_evaluation_r75 import reference_quality


def key(row):
    return row['seed'], row['scene'], row['label']


def main():
    verify()
    rows, sources = reference_quality()
    new = []
    for seed in (0, 1):
        root = BASE/f'results/exploration-r82-eval-seed{seed}'
        terminal = json.loads((root/'runtime.json').read_text())
        assert terminal['completed_normally'] and terminal['quality_rows'] == 50 and terminal['frame_rows'] == 2240
        path = root/'per_sequence.jsonl'; values = lines(path)
        assert len(values) == 50
        new += values; sources[str(path.relative_to(ROOT))] = digest(path)
    root = BASE/'results/exploration-r82-perceptual001'
    terminal = json.loads((root/'runtime.json').read_text())
    assert terminal['completed_normally'] and terminal['quality_rows'] == 100 and terminal['comparison_frames'] == 4480
    path = root/'per_sequence.jsonl'; perceptual = lines(path)
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
    detail = {key(r): r for r in rows}; gates = []
    controls = ('MobileIE_raw', 'spatial', 'direct', 'centered_wrong',
                'MobileIE_recursive_self05', 'spatial_recursive_self05')
    for domain in ('davis_train_pilot', 'paired_real_ND_DEV'):
        for seed in (0, 1):
            candidate = index[domain, seed, 'centered']
            for label in controls:
                ref = index[domain, seed, label]
                psnr = candidate['psnr_rgb']-ref['psnr_rgb']
                jod = candidate['JOD']-ref['JOD']
                ratio = candidate['temporal_error']/ref['temporal_error']
                strong = label in controls[-2:]
                deltas = [dict(scene=r['scene'], PSNR_delta=r['psnr_rgb']-detail[seed,r['scene'],label]['psnr_rgb'],
                    JOD_delta=r['JOD']-detail[seed,r['scene'],label]['JOD'])
                    for r in rows if r['domain'] == domain and r['seed'] == seed and r['label'] == 'centered']
                worst = min(r['PSNR_delta'] for r in deltas)
                quality = psnr >= -.05 and jod >= -.02 and (not strong or worst >= -.5)
                gates.append(dict(domain=domain, seed=seed, reference=label, PSNR_delta=psnr, JOD_delta=jod,
                    Etemp_ratio=ratio, Etemp_max_A=.90 if strong else .95,
                    Etemp_max_B=1.05 if strong else .95, worst_condition_PSNR_delta=worst,
                    condition_deltas=deltas, strong_recursive_reference=strong,
                    passed_A_quality=quality and ratio <= (.90 if strong else .95),
                    passed_B_quality=quality and ratio <= (1.05 if strong else .95)))
    assert len(gates) == 24
    out = BASE/'report/exploration-r82-joint'; out.mkdir(exist_ok=False)
    (out/'all_conditions.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows))
    write(out/'domain_seed_means.json', means); write(out/'gates.json', gates); write(out/'sources.json', sources)
    table = ['# R82固定共同条件修正完整表', '',
        '重复DEV：六SYN条件来自三源、四ND源。F0固定4075参数/8000原更新，三H各304参数/8000新更新，合4379参数；centered双H，非零额外推理开销。错误历史仅外源teacher-forced诊断。', '',
        '|域|seed|控制|PSNR|Etemp|JOD|', '|---|---:|---|---:|---:|---:|']
    for r in means:
        table.append(f"|{r['domain']}|{r['seed']}|{r['label']}|{r['psnr_rgb']:.6f}|{r['temporal_error']:.6f}|{r['JOD']:.6f}|")
    (out/'tables.md').write_text('\n'.join(table)+'\n')
    a = all(r['passed_A_quality'] for r in gates)
    b = all(r['passed_B_quality'] for r in gates)
    write(out/'audit.json', dict(completed=True, new_quality_rows=100, reused_reference_rows=40,
        all_quality_rows=140, domain_seed_means=28, comparisons=24,
        A_quality_eligible=a, B_quality_eligible=b, cost_measurement_eligible=a or b,
        A_cost_median_ratio_max=1.05, B_cost_median_ratio_max=.80,
        full_deployment_cost_measured=False, full_A_or_B_pass_not_established=True,
        formal_optimizer_updates=48000,
        decision='Full CPU cost qualification pending' if a or b else 'STOP this H local grid',
        new_method_established=False, novelty_established=False, natural_night_claim=False,
        independent_final=False, phone_measured=False))


if __name__ == '__main__':
    main()
