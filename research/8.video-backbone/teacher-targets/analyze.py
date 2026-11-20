"""Descriptive complete-table differences, after the unchanged primary decision."""
import json
from statistics import mean
from study_io import ROOT, BASE, read, write, stamp, digest, verify

METRICS = ('psnr_rgb', 'ssim_rgb', 'temporal_error', 'JOD')


def main():
    verify()
    source = BASE / 'report/joint/all_conditions.jsonl'
    audit = read(BASE / 'report/joint/audit.json')
    assert audit['completed'] and audit['conditions'] == 240
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    labels = sorted({r['label'] for r in rows})
    groups = sorted({(r['domain'], r['seed']) for r in rows})
    values = {(r['seed'], r['scene'], r['label']): r for r in rows}
    pairs = [('t1_temporal_raw', label) for label in labels if label != 't1_temporal_raw']
    pairs += [(f't1_{arm}_raw', 'MobileIE_raw') for arm in ('gt', 'spatial')]
    pairs += [(f't1_temporal_{method}05', ref) for method in ('FB', 'DIS') for ref in
              (('MobileIE_recursive_self05' if method == 'FB' else 'DIS_recursive_self05'),
               f't1_gt_{method}05', f't1_spatial_{method}05')]
    assert len(pairs) == len(set(pairs)) == 19
    conditions, summaries = [], []
    for candidate, reference in pairs:
        for domain, seed in groups:
            chosen = [r for r in rows if (r['domain'], r['seed'], r['label']) == (domain, seed, candidate)]
            assert len(chosen) == (6 if domain == 'davis_train_pilot' else 4)
            local = []
            for row in chosen:
                ref = values[seed, row['scene'], reference]
                diff = {m + '_difference': row[m] - ref[m] for m in METRICS}
                item = dict(domain=domain, seed=seed, scene=row['scene'], candidate=candidate,
                            reference=reference, **diff, temporal_ratio=row['temporal_error'] / ref['temporal_error'],
                            joint_nonworse=diff['psnr_rgb_difference'] >= 0 and diff['JOD_difference'] >= 0
                            and diff['temporal_error_difference'] <= 0)
                conditions.append(item)
                local.append(item)
            candidate_means = {m: mean(r[m] for r in chosen) for m in METRICS}
            reference_means = {m: mean(values[seed, r['scene'], reference][m] for r in chosen) for m in METRICS}
            summaries.append(dict(domain=domain, seed=seed, candidate=candidate, reference=reference,
                **{m + '_difference': candidate_means[m] - reference_means[m] for m in METRICS},
                temporal_ratio=candidate_means['temporal_error'] / reference_means['temporal_error'],
                worst_condition_PSNR_difference=min(r['psnr_rgb_difference'] for r in local),
                worst_condition_JOD_difference=min(r['JOD_difference'] for r in local),
                temporal_better_conditions=sum(r['temporal_error_difference'] < 0 for r in local),
                PSNR_better_conditions=sum(r['psnr_rgb_difference'] > 0 for r in local),
                joint_nonworse_conditions=sum(r['joint_nonworse'] for r in local), conditions=len(local)))
    assert (len(conditions), len(summaries)) == (380, 76)
    out = BASE / 'report/analysis'
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'condition_differences.jsonl').open('x') as stream:
        for row in conditions:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    write(out / 'group_differences.json', summaries)
    write(out / 'audit.json', dict(at=stamp(), completed=True, descriptive_only=True,
          source=str(source.relative_to(ROOT)), source_sha256=digest(source),
          analyzer_sha256=digest(__file__), compared_pairs=19, condition_differences=380,
          group_differences=76, primary_decision_changed=False, new_model_forward_images=0,
          new_quality_or_JOD_evaluations=0, independent_final_opened=False))
    table = ['# 完整T1表的描述性差分', '', '事后完整差分，不增加或重定义原主要门槛；全部逐条件正例保留。', '',
             '|域|seed|候选|参考|ΔPSNR|ΔJOD|Etemp比|时间改善条件|三指标共同不劣条件|',
             '|---|---:|---|---|---:|---:|---:|---:|---:|']
    for row in summaries:
        table.append(f'|{row["domain"]}|{row["seed"]}|{row["candidate"]}|{row["reference"]}|'
            f'{row["psnr_rgb_difference"]:+.6f}|{row["JOD_difference"]:+.6f}|{row["temporal_ratio"]:.6f}|'
            f'{row["temporal_better_conditions"]}/{row["conditions"]}|{row["joint_nonworse_conditions"]}/{row["conditions"]}|')
    (out / 'differences.md').write_text('\n'.join(table) + '\n')


if __name__ == '__main__':
    main()
