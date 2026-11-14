"""Close matched FP32 training-form quality; keep precision history separate."""
from collections import defaultdict
import json
from statistics import mean

from dynamic_training_r34 import BASE, ROOT, digest

METRICS = ['psnr_rgb', 'ssim_rgb', 'temporal_error', 'edge_mae']
SYNTHETIC, REAL = 'davis_train_pilot', 'paired_real_ND_DEV'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    rows, sources = [], {}
    for seed in [0, 1]:
        folder = BASE/f'results/exploration-r41-stage002-eval-seed{seed}'
        runtime = json.loads((folder/'runtime.json').read_text())
        assert runtime['completed_normally'] and runtime['rows'] == 60
        assert runtime['neural_sequences'] == 20 and not runtime['cuda_initialized']
        values = list(map(json.loads, (folder/'per_sequence.jsonl').read_text().splitlines()))
        assert len(values) == 60 and {r['train_seed'] for r in values} == {seed}
        rows += values
        for name in ['config.json', 'runtime.json', 'per_sequence.jsonl']:
            path = folder/name
            sources[str(path.relative_to(ROOT))] = digest(path)
    assert len(rows) == 120
    lookup = {(r['domain'], r['model'], r['scene'], r['train_seed'], r['method']): r for r in rows}
    assert len(lookup) == 120
    groups, seeds, domains, pairs = defaultdict(list), [], [], []
    for row in rows:
        groups[(row['domain'], row['anchor_family'], row['train_seed'], row['method'])].append(row)
    for (domain, family, seed, method), values in sorted(groups.items()):
        assert len(values) == (6 if domain == SYNTHETIC else 4)
        seeds.append({'domain': domain, 'anchor_family': family, 'seed': seed, 'method': method,
                      **{m: mean(r[m] for r in values) for m in METRICS}})
    old_report = BASE/'report/exploration-r40-mixed-anchor'
    old_audit = json.loads((old_report/'audit.json').read_text())
    assert old_audit['passed'] and old_audit['new_rows'] == 60 and old_audit['total_rows'] == 156
    old_domains = json.loads((old_report/'domain_summary.json').read_text())
    assert len(old_domains) == 18
    old_seeds = json.loads((old_report/'domain_seed_summary.json').read_text())
    seeds += [{**r, 'reference_source': 'R40 complete legacy summaries; not matched FP32 primary control'} for r in old_seeds]
    for name in ['audit.json', 'domain_seed_summary.json', 'domain_summary.json']:
        path = old_report/name
        sources[str(path.relative_to(ROOT))] = digest(path)
    groups = defaultdict(list)
    for row in seeds:
        groups[(row['domain'], row['anchor_family'], row['method'])].append(row)
    for (domain, family, method), values in sorted(groups.items()):
        assert {r['seed'] for r in values} == ({None} if family == 'official_LOL' else {0, 1})
        domains.append({'domain': domain, 'anchor_family': family, 'method': method,
                        'source_videos': 3 if domain == SYNTHETIC else 4,
                        'input_conditions': 6 if domain == SYNTHETIC else 4,
                        'weight_instances': len(values),
                        'training_compute': 'strict FP32' if family.startswith('R41_') else 'legacy, see source configuration',
                        **{m: mean(r[m] for r in values) for m in METRICS}})
    assert len(domains) == 30
    effects = []
    for domain in [SYNTHETIC, REAL]:
        for seed in [0, 1]:
            def select(family):
                return next(r for r in seeds if r['domain'] == domain and r['anchor_family'] == family
                            and r['seed'] == seed and r['method'] == 'raw')
            full, slim, old = select('R41_expanded_fp32'), select('R41_slim_fp32'), select('R40_mixed')
            effects.append({'domain': domain, 'seed': seed,
                            **{'expanded_minus_matched_slim_'+m: full[m]-slim[m] for m in METRICS},
                            'matched_slim_minus_legacy_R40_psnr': slim['psnr_rgb']-old['psnr_rgb'],
                            'expanded_minus_legacy_R40_psnr': full['psnr_rgb']-old['psnr_rgb']})
    winning_domains = []
    for domain in [SYNTHETIC, REAL]:
        gain = [r['expanded_minus_matched_slim_psnr_rgb'] for r in effects if r['domain'] == domain]
        other = [r['expanded_minus_matched_slim_psnr_rgb'] for r in effects if r['domain'] != domain]
        if all(value >= .2 for value in gain) and all(value >= -.05 for value in other):
            winning_domains.append(domain)
    old_lookup = {}
    for seed in [0, 1]:
        path = BASE/f'results/exploration-r40-eval-seed{seed}/per_sequence.jsonl'
        for row in map(json.loads, path.read_text().splitlines()):
            old_lookup[(row['domain'], row['scene'], seed, row['method'])] = row
        sources[str(path.relative_to(ROOT))] = digest(path)
    assert len(old_lookup) == 60
    for row in rows:
        references = [('legacy_R40', old_lookup[(row['domain'], row['scene'], row['train_seed'], row['method'])])]
        if row['model'] == 'expanded_fp32':
            references.append(('matched_slim_fp32', lookup[(row['domain'], 'slim_fp32', row['scene'], row['train_seed'], row['method'])]))
        for reference, ref in references:
            pairs.append({'domain': row['domain'], 'scene': row['scene'], 'seed': row['train_seed'],
                          'model': row['model'], 'method': row['method'], 'reference': reference,
                          **{'delta_'+m: row[m]-ref[m] for m in METRICS}})
    out = BASE/'report/exploration-r41-training-form'
    out.mkdir(parents=True, exist_ok=False)
    for name, value in [('domain_summary.json', domains), ('domain_seed_summary.json', seeds),
                        ('paired_by_condition.json', pairs), ('raw_matched_effects.json', effects)]:
        write(out/name, value)
    write(out/'audit.json', {'passed': True, 'stage': 'stage002', 'new_rows': 120, 'new_neural_sequences': 40,
          'legacy_domain_summary_rows': 18, 'primary_winning_domains': winning_domains,
          'matched_baseline_screening_passed': bool(winning_domains), 'source_sha256': sources,
          'scope': 'Original training-form control under common FP32; prior partial updates and failures retained separately. No novelty, actual history or final-test inference.'})
    lines = ['# R41 strict-FP32 training-form control', '',
             'Corrected stage002, final4000 only, two training forms × two seeds. Both train under strict FP32 with the same windows, optimizer and update count. Expanded raw convolution executes the preserved outputs separately, without adding parameters; this numerical implementation differs from the original single wide-kernel call. Both deployments are the same 4075-parameter slim class.', '',
             'The primary comparison is expanded versus this stage’s matched slim FP32. R40 and other legacy references remain separate: their training precision was not this matched control. Equal updates/images do not imply equal training FLOPs. No warm-up, official full-schedule reproduction, video innovation, phone cost or independent test is claimed.', '']
    for domain in [SYNTHETIC, REAL]:
        lines += ['## '+domain, '', '|Anchor family|Output|PSNR|SSIM|Temporal↓|Edge↓|', '|---|---|---:|---:|---:|---:|']
        for row in domains:
            if row['domain'] == domain:
                lines.append('|'+row['anchor_family']+'|'+row['method']+'|'+'|'.join(f'{row[m]:.6f}' for m in METRICS)+'|')
        lines.append('')
    lines += ['|Domain|Seed|Expanded ΔPSNR vs matched slim|Matched slim ΔPSNR vs old R40|', '|---|---:|---:|---:|']
    for row in effects:
        lines.append(f"|{row['domain']}|{row['seed']}|{row['expanded_minus_matched_slim_psnr_rgb']:.6f}|{row['matched_slim_minus_legacy_R40_psnr']:.6f}|")
    lines += ['', 'Registered same-domain, both-seed baseline screen: '+str(bool(winning_domains))+'; qualifying domains: '+str(winning_domains)+'. Both seeds must gain .2dB in the same domain and lose no more than .05dB in the other. This is a baseline screen, not temporal-method validation.', '',
              'All raw/finite/recursive results and pairs remain visible. Finite/recursive use own-raw half-FB and explicit RGB history, without extra teacher or GT reset. Four real sources are reused daytime ND DEV; synthetic conditions are three source videos, not six independent benchmarks. Initial numeric gates do not establish bitwise-equivalent training.']
    (out/'tables.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'new_rows': 120, 'primary_winning_domains': winning_domains}))


if __name__ == '__main__':
    main()
