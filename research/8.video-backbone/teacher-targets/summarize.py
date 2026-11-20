"""Complete joint DEV table and registered primary quality gates, final-only."""
import json
from statistics import mean

from study_io import ROOT, BASE, PROCESS, write, digest, stamp, verify

METRICS = ('psnr_rgb', 'ssim_rgb', 'temporal_error', 'JOD')
RAW_CONTROLS = ('t1_gt_raw', 't1_spatial_raw')
REFERENCES = ('MobileIE_recursive_self05', 'DIS_recursive_self05',
              *(f't1_{arm}_{method}05' for arm in ('gt', 'spatial', 'temporal')
                for method in ('FB', 'DIS')), *RAW_CONTROLS)
assert len(REFERENCES) == len(set(REFERENCES)) == 10


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def key(row):
    return row['seed'], row['scene'], row['label']


def main():
    verify()
    sources = {}
    def read(path):
        sources[str(path.relative_to(ROOT))] = digest(path)
        return lines(path)
    rows = []
    for seed in (0, 1):
        quality = BASE / f'results/quality-seed{seed}'
        assert json.loads((quality / 'runtime.json').read_text())['completed_normally']
        rows += read(quality / 'per_sequence.jsonl')
    assert len(rows) == len({key(r) for r in rows}) == 180
    assert len(REFERENCES) == 10
    attempts = sorted((BASE / 'results').glob('perceptual-attempt*/runtime.json'))
    metric = json.loads(attempts[-1].read_text())
    assert metric['completed_normally'] and (metric['quality_rows'], metric['comparison_frames']) == (180, 8064)
    perception = {key(r): r for r in read(attempts[-1].parent / 'per_sequence.jsonl')}
    assert set(perception) == {key(r) for r in rows}
    for row in rows:
        row['JOD'] = perception[key(row)]['JOD']
        assert row['parameters'] == 4075
    references = read(ROOT / 'research/4.temporal-consistency/report/exploration-r86-joint/all_conditions.jsonl')
    assert len(references) == 60 and all('JOD' in r for r in references)
    rows += references
    assert len(rows) == len({key(r) for r in rows}) == 240
    groups = sorted({(r['domain'], r['seed']) for r in rows})
    labels = sorted({r['label'] for r in rows})
    assert len(groups) == 4 and len(labels) == 12
    means = []
    for domain, seed in groups:
        for label in labels:
            selected = [r for r in rows if (r['domain'], r['seed'], r['label']) == (domain, seed, label)]
            assert len(selected) == (6 if domain == 'davis_train_pilot' else 4)
            means.append(dict(domain=domain, seed=seed, label=label,
                              **{metric: mean(r[metric] for r in selected) for metric in METRICS}))
    lookup = {(r['domain'], r['seed'], r['label']): r for r in means}
    conditions = {key(r): r for r in rows}
    protections, comparisons = [], []
    for domain, seed in groups:
        spatial = lookup[domain, seed, 't1_temporal_raw']
        f0 = lookup[domain, seed, 'MobileIE_raw']
        scenes = [r['scene'] for r in rows if (r['domain'], r['seed'], r['label']) == (domain, seed, 'MobileIE_raw')]
        worst_spatial = min(conditions[seed, s, 't1_temporal_raw']['psnr_rgb'] -
                            conditions[seed, s, 'MobileIE_raw']['psnr_rgb'] for s in scenes)
        protections.append(dict(domain=domain, seed=seed, PSNR_difference=spatial['psnr_rgb'] - f0['psnr_rgb'],
            JOD_difference=spatial['JOD'] - f0['JOD'], worst_condition_PSNR_difference=worst_spatial,
            passed=(spatial['psnr_rgb'] >= f0['psnr_rgb'] - .10 and spatial['JOD'] >= f0['JOD'] - .05
                    and worst_spatial >= -.5)))
        candidate = lookup[domain, seed, 't1_temporal_raw']
        for reference in REFERENCES:
            ref = lookup[domain, seed, reference]
            worst_psnr = min(conditions[seed, s, 't1_temporal_raw']['psnr_rgb'] -
                             conditions[seed, s, reference]['psnr_rgb'] for s in scenes)
            worst_jod = min(conditions[seed, s, 't1_temporal_raw']['JOD'] -
                            conditions[seed, s, reference]['JOD'] for s in scenes)
            quality = (candidate['psnr_rgb'] >= ref['psnr_rgb'] - .05 and
                       candidate['JOD'] >= ref['JOD'] - .03 and worst_psnr >= -.5 and worst_jod >= -.15)
            ratio = candidate['temporal_error'] / ref['temporal_error']
            comparisons.append(dict(domain=domain, seed=seed, reference=reference,
                PSNR_difference=candidate['psnr_rgb'] - ref['psnr_rgb'],
                JOD_difference=candidate['JOD'] - ref['JOD'], temporal_error_ratio=ratio,
                worst_condition_PSNR_difference=worst_psnr, worst_condition_JOD_difference=worst_jod,
                quality_protected=quality, A_pair=quality and ratio <= .90,
                B_pair=quality and ratio <= ( .95 if reference in RAW_CONTROLS else 1.02)))
    assert len(protections) == 4 and len(comparisons) == 40
    A = all(p['passed'] for p in protections) and all(p['A_pair'] for p in comparisons)
    B = all(p['passed'] for p in protections) and all(p['B_pair'] for p in comparisons)
    out = BASE / 'report/joint'
    out.mkdir(parents=True, exist_ok=False)
    for filename, value in [('domain_seed_means.json', means), ('recovery_protection.json', protections),
                            ('primary_comparisons.json', comparisons), ('sources.json', sources)]:
        write(out / filename, value)
    with (out / 'all_conditions.jsonl').open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    text = ['# T1同F0已知蒸馏对照完整开发表', '', '重复DEV；普通KD对照，不主张方法创新；自然动态、手机和独立最终仍待证。', '',
            '|域|seed|输出|PSNR|SSIM|Etemp|JOD|', '|---|---:|---|---:|---:|---:|---:|']
    for row in means:
        text.append(f'|{row["domain"]}|{row["seed"]}|{row["label"]}|' + '|'.join(f'{row[m]:.6f}' for m in METRICS) + '|')
    text += ['', f'原F0恢复保护 {sum(p["passed"] for p in protections)}/4；'
             f'主要A比较 {sum(p["A_pair"] for p in comparisons)}/40，'
             f'B比较 {sum(p["B_pair"] for p in comparisons)}/40。', '',
             f'完整A质量资格={A}；B质量资格={B}。只有任一True才进入预定完整成本；这是质量资格，不是方法或速度成功。']
    (out / 'tables.md').write_text('\n'.join(text) + '\n')
    write(out / 'audit.json', dict(at=stamp(), completed=True, conditions=240, means=len(means),
          new_quality_conditions=180, new_quality_frames=8064, reused_reference_conditions=60,
          independent_videos=False, known_KD_control=True, A_quality_eligible=A, B_quality_eligible=B,
          cost_extension_eligible=A or B, cost_measured=False, phone_measured=False,
          innovation_established=False, independent_final_opened=False))


if __name__ == '__main__':
    main()
