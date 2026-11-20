"""Describe all completed V1 paired outcomes; never change registered gates."""
import hashlib
import json
from datetime import datetime
from pathlib import Path


BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
PAIRS = (
    ('spatial/F0', 'topic8_spatial_raw', 'MobileIE_raw'),
    ('learned/F0-FB', 'topic8_learned_causal', 'MobileIE_recursive_self05'),
    ('learned/F0-DIS', 'topic8_learned_causal', 'DIS_recursive_self05'),
    ('learned/own-preview', 'topic8_learned_causal', 'topic8_learned_preview'),
    ('learned/own-FB', 'topic8_learned_causal', 'topic8_learned_preview_FB05'),
    ('learned/own-DIS', 'topic8_learned_causal', 'topic8_learned_preview_DIS05'),
    ('learned/fixed', 'topic8_learned_causal', 'topic8_fixed_causal'),
    ('fixed/own-preview', 'topic8_fixed_causal', 'topic8_fixed_preview'),
    ('fixed/own-FB', 'topic8_fixed_causal', 'topic8_fixed_preview_FB05'),
    ('fixed/own-DIS', 'topic8_fixed_causal', 'topic8_fixed_preview_DIS05'),
    ('fixed/F0-FB', 'topic8_fixed_causal', 'MobileIE_recursive_self05'),
    ('fixed/F0-DIS', 'topic8_fixed_causal', 'DIS_recursive_self05'),
    ('spatial-FB/F0-FB', 'topic8_spatial_raw_FB05', 'MobileIE_recursive_self05'),
    ('spatial-DIS/F0-DIS', 'topic8_spatial_raw_DIS05', 'DIS_recursive_self05'),
    ('learned/spatial-FB', 'topic8_learned_causal', 'topic8_spatial_raw_FB05'),
    ('learned/spatial-DIS', 'topic8_learned_causal', 'topic8_spatial_raw_DIS05'),
)


def read(path):
    return json.loads(path.read_text())


def difference(candidate, reference):
    return dict(PSNR_difference=candidate['psnr_rgb'] - reference['psnr_rgb'],
                JOD_difference=candidate['JOD'] - reference['JOD'],
                temporal_error_change_percent=100 *
                (candidate['temporal_error'] / reference['temporal_error'] - 1))


def main():
    joint = BASE / 'report/v1-joint'
    audit = read(joint / 'audit.json')
    assert audit['completed'] and audit['conditions'] == 280 and audit['means'] == 56
    rows = [json.loads(line) for line in (joint / 'all_conditions.jsonl').read_text().splitlines()]
    lookup = {(r['seed'], r['scene'], r['label']): r for r in rows}
    means = read(joint / 'domain_seed_means.json')
    groups = {(r['domain'], r['seed'], r['label']): r for r in means}
    conditions = sorted((r['seed'], r['scene']) for r in rows if r['label'] == 'MobileIE_raw')
    assert len(lookup) == 280 and len(groups) == 56 and len(conditions) == 20
    summaries, all_conditions, all_means = [], [], []
    for name, candidate, reference in PAIRS:
        paired, grouped = [], []
        for seed, scene in conditions:
            c, r = lookup[seed, scene, candidate], lookup[seed, scene, reference]
            assert c['domain'] == r['domain']
            paired.append(dict(pair=name, candidate=candidate, reference=reference,
                               seed=seed, scene=scene, domain=c['domain'], **difference(c, r)))
        for domain, seed in sorted({(r['domain'], r['seed']) for r in means}):
            grouped.append(dict(pair=name, candidate=candidate, reference=reference,
                                domain=domain, seed=seed,
                                **difference(groups[domain, seed, candidate], groups[domain, seed, reference])))
        assert len(paired) == 20 and len(grouped) == 4
        summaries.append(dict(pair=name, candidate=candidate, reference=reference,
            domain_seed_ranges={key: [min(r[key] for r in grouped), max(r[key] for r in grouped)]
                                for key in ('PSNR_difference', 'JOD_difference', 'temporal_error_change_percent')},
            condition_counts=dict(total=20,
                PSNR_positive=sum(r['PSNR_difference'] > 0 for r in paired),
                JOD_positive=sum(r['JOD_difference'] > 0 for r in paired),
                temporal_error_lower=sum(r['temporal_error_change_percent'] < 0 for r in paired),
                strict_joint_nonworse=sum(r['PSNR_difference'] >= 0 and r['JOD_difference'] >= 0 and
                                         r['temporal_error_change_percent'] <= 0 for r in paired))))
        all_conditions += paired
        all_means += grouped
    out = BASE / 'report/v1-analysis'
    out.mkdir(parents=True, exist_ok=False)
    files = [joint / 'audit.json', joint / 'domain_seed_means.json', joint / 'all_conditions.jsonl', Path(__file__)]
    result = dict(at=datetime.now().astimezone().isoformat(),
        sources={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        descriptive_posthoc=True, registered_gates_changed=False, independent_videos=False,
        new_model_forwards=0, new_metric_calls=0, summaries=summaries,
        all_domain_seed_differences=all_means, all_condition_differences=all_conditions)
    (out / 'paired_outcomes.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    text = ['# 全部V1成对结果', '',
            '仅完整重复DEV的描述分析；不修改注册门槛，不作置信区间或独立视频推断。', '',
            '范围跨四个域×seed均值；条件计数覆盖全部20项。Et为相对变化百分数，负数较好。', '',
            '|候选/参考|PSNR差范围|JOD差范围|Et变化范围%|PSNR正/20|JOD正/20|Et下降/20|三者同时不差/20|',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in summaries:
        ranges, counts = row['domain_seed_ranges'], row['condition_counts']
        formatted = [' / '.join(f'{v:+.5f}' for v in ranges[k]) for k in
                     ('PSNR_difference', 'JOD_difference', 'temporal_error_change_percent')]
        text.append('|' + '|'.join([row['pair'], *formatted, str(counts['PSNR_positive']),
                    str(counts['JOD_positive']), str(counts['temporal_error_lower']),
                    str(counts['strict_joint_nonworse'])]) + '|')
    (out / 'tables.md').write_text('\n'.join(text) + '\n')


if __name__ == '__main__':
    main()
