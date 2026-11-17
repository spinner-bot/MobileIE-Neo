"""All fixed classical-reference comparisons; no threshold or seed selection."""
import json
import numpy as np

from dis_reference_io_r86 import BASE, QUALITY, COST, METRIC, LABEL, verify, lines, write, stamp, digest
from causal_reference_evaluation_r75 import reference_quality


def main():
    verify()
    for root, count_key, expected in [(QUALITY, 'quality_rows', 20), (METRIC, 'quality_rows', 20),
                                      (COST, 'measured_sequences', 120)]:
        runtime = json.loads((root/'runtime.json').read_text())
        assert runtime['completed_normally'] and runtime[count_key] == expected
    quality, metric = lines(QUALITY/'per_sequence.jsonl'), lines(METRIC/'per_sequence.jsonl')
    scores = {(r['seed'], r['scene'], r['label']): r for r in metric}
    assert len(scores) == len(quality) == 20
    old, old_sources = reference_quality()
    combined = old+[dict(r, JOD=scores[r['seed'], r['scene'], r['label']]['JOD']) for r in quality]
    lookup = {(r['seed'], r['scene'], r['label']): r for r in combined}
    assert len(lookup) == len(combined) == 60
    cost = json.loads((COST/'summary.json').read_text())
    cost_groups = {(g['domain'], g['seed']): g for g in cost['groups']}
    means, gates = [], []
    for domain in ('davis_train_pilot', 'paired_real_ND_DEV'):
        for seed in (0, 1):
            grouped = {}
            for label in ('MobileIE_raw', 'MobileIE_recursive_self05', LABEL):
                selected = [r for r in combined if (r['domain'], r['seed'], r['label']) == (domain, seed, label)]
                assert len(selected) == (6 if domain == 'davis_train_pilot' else 4)
                row = dict(domain=domain, seed=seed, label=label, conditions=len(selected),
                    **{k: float(np.mean([r[k] for r in selected])) for k in ('psnr_rgb', 'temporal_error', 'JOD')})
                means.append(row); grouped[label] = row
            new, ref = grouped[LABEL], grouped['MobileIE_recursive_self05']
            condition_deltas = []
            for row in [r for r in combined if (r['domain'], r['seed'], r['label']) == (domain, seed, LABEL)]:
                previous = lookup[seed, row['scene'], 'MobileIE_recursive_self05']
                condition_deltas.append(dict(scene=row['scene'], PSNR_delta=row['psnr_rgb']-previous['psnr_rgb'],
                    JOD_delta=row['JOD']-previous['JOD'], Etemp_ratio=row['temporal_error']/previous['temporal_error']))
            psnr_delta, jod_delta = new['psnr_rgb']-ref['psnr_rgb'], new['JOD']-ref['JOD']
            ratio = new['temporal_error']/ref['temporal_error']
            worst_psnr = min(r['PSNR_delta'] for r in condition_deltas)
            worst_jod = min(r['JOD_delta'] for r in condition_deltas)
            tests = dict(PSNR=psnr_delta >= -.05, JOD=jod_delta >= -.02, Etemp=ratio <= 1.05,
                         worst_condition_PSNR=worst_psnr >= -.30, worst_condition_JOD=worst_jod >= -.10)
            speed = cost_groups[domain, seed]
            gates.append(dict(domain=domain, seed=seed, PSNR_delta=psnr_delta, JOD_delta=jod_delta,
                Etemp_ratio=ratio, worst_condition_PSNR_delta=worst_psnr, worst_condition_JOD_delta=worst_jod,
                tests=tests, quality_gate=all(tests.values()), speed=speed, speed_gate=speed['speed_gate'],
                joint_gate=all(tests.values()) and speed['speed_gate'], conditions=condition_deltas))
    out = BASE/'report/exploration-r86-joint'; out.mkdir(exist_ok=False)
    for row in combined:
        with (out/'all_conditions.jsonl').open('a') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    write(out/'domain_seed_means.json', means); write(out/'gates.json', gates)
    table = ['# Fixed native DIS reference: complete quality and same-run CPU costs', '',
        'Repeated DEV, both seeds. Known classical preset, not an innovative method, handset or independent final.', '',
        '|Domain|Seed|Output|PSNR|Etemp|JOD|', '|---|---:|---|---:|---:|---:|']
    for row in means:
        table.append(f"|{row['domain']}|{row['seed']}|{row['label']}|{row['psnr_rgb']:.6f}|{row['temporal_error']:.6f}|{row['JOD']:.6f}|")
    table += ['', '|Domain|Seed|DIS−FB PSNR|DIS/FB Etemp|DIS−FB JOD|DIS ms|FB ms|Mean paired cost ratio|Quality/speed/joint|',
              '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for row in gates:
        c = row['speed']
        table.append(f"|{row['domain']}|{row['seed']}|{row['PSNR_delta']:+.6f}|{row['Etemp_ratio']:.6f}|"
            f"{row['JOD_delta']:+.6f}|{c['mean_condition_median_DIS_ms']:.3f}|{c['mean_condition_median_half_FB_ms']:.3f}|"
            f"{c['mean_condition_ratio']:.6f}|{row['quality_gate']}/{row['speed_gate']}/{row['joint_gate']}|")
    table += ['', 'Quality uses SYN64/ND16 frames; cost uses the same-source first16, both complete pipelines in the same new run, '
        'two warmups and three measured repeats, mean across matched condition medians. Shared CPU, no camera/display/file I/O. '
        'All per-condition quality losses and cost repeats remain in raw files; no best-seed selection or cross-run R85 speed ratio.']
    (out/'tables.md').write_text('\n'.join(table)+'\n')
    sources = {str(p.relative_to(BASE.parents[1])): digest(p) for p in
        [QUALITY/'per_sequence.jsonl', METRIC/'per_sequence.jsonl', COST/'summary.json']}
    sources.update(old_sources)
    write(out/'audit.json', dict(at=stamp(), completed=True, new_quality_rows=20, reused_reference_rows=40,
        all_quality_rows=60, means=12, comparisons=4, quality_passed=sum(r['quality_gate'] for r in gates),
        speed_passed=sum(r['speed_gate'] for r in gates), joint_passed=sum(r['joint_gate'] for r in gates),
        all_joint_passed=all(r['joint_gate'] for r in gates), new_SGD=0, new_method_established=False,
        handset_measured=False, independent_final=False, sources=sources))
    print(json.dumps(dict(completed=True, joint_passed=sum(r['joint_gate'] for r in gates), comparisons=4)))


if __name__ == '__main__':
    main()
