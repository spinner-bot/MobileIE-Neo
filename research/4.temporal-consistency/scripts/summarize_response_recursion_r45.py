"""Separate independent latent-cue evidence from generic learned fusion and alpha tuning."""
from collections import defaultdict
import json
from statistics import mean

from dynamic_training_r34 import BASE, ROOT, digest
from evaluate_response_recursion_r45 import ALPHAS, METHODS

METRICS = ['psnr_rgb', 'ssim_rgb', 'temporal_error', 'edge_mae']
COMPARISONS = [('response_vs_scalar', 'response', 'scalar'),
               ('response_vs_rgb', 'response', 'rgb'),
               ('response_vs_fixed05', 'response', 'fixed_05'),
               ('rgb_vs_scalar', 'rgb', 'scalar'),
               ('scalar_vs_fixed05', 'scalar', 'fixed_05'),
               ('rgb_vs_fixed05', 'rgb', 'fixed_05')]


def main():
    rows, stats, sources = [], [], {}
    for seed in [0, 1]:
        folder = BASE/f'results/exploration-r45-stage001-eval-seed{seed}'
        runtime = json.loads((folder/'runtime.json').read_text())
        assert runtime['completed_normally'] and runtime['rows'] == 90 and runtime['neural_sequences'] == 10
        assert not runtime['cuda_initialized'] and runtime['raw_replay_max_abs_normalized'] <= 1e-5
        current = [json.loads(l) for l in (folder/'per_sequence.jsonl').read_text().splitlines()]
        assert len(current) == 90 and {r['train_seed'] for r in current} == {seed}
        rows += current
        values = [json.loads(l) for l in (folder/'fusion_stats.jsonl').read_text().splitlines()]
        assert len(values) == 80
        stats += values
        for p in [folder/'config.json', folder/'runtime.json', folder/'per_sequence.jsonl', folder/'fusion_stats.jsonl']:
            sources[str(p.relative_to(ROOT))] = digest(p)
    lookup = {(r['domain'], r['scene'], r['train_seed'], r['method']): r for r in rows}
    assert len(rows) == len(lookup) == 180
    conditions = sorted({key[:3] for key in lookup})
    assert len(conditions) == 20
    groups, seeds = defaultdict(list), []
    for r in rows:
        groups[(r['domain'], r['train_seed'], r['method'])].append(r)
    for (domain, seed, method), values in sorted(groups.items()):
        assert len(values) == (6 if domain == 'davis_train_pilot' else 4)
        seeds.append({'domain': domain, 'seed': seed, 'method': method,
                      **{m: mean(r[m] for r in values) for m in METRICS}})
    assert len(seeds) == 36
    means = {(r['domain'], r['seed'], r['method']): r for r in seeds}
    pairs, effects, gates = [], [], {}
    domain_seeds = sorted({key[:2] for key in means})
    for name, model, reference in COMPARISONS:
        local, summaries = [], []
        for domain, scene, seed in conditions:
            a, b = [lookup[(domain, scene, seed, method)] for method in [model, reference]]
            assert b['temporal_error'] > 0
            r = {'comparison': name, 'domain': domain, 'scene': scene, 'seed': seed,
                 **{'delta_'+m: a[m]-b[m] for m in METRICS},
                 'temporal_ratio': a['temporal_error']/b['temporal_error']}
            pairs.append(r);local.append(r)
        for domain, seed in domain_seeds:
            a, b = [means[(domain, seed, method)] for method in [model, reference]]
            r = {'comparison': name, 'domain': domain, 'seed': seed,
                 **{'delta_'+m: a[m]-b[m] for m in METRICS},
                 'temporal_ratio': a['temporal_error']/b['temporal_error']}
            effects.append(r);summaries.append(r)
        quality = all(r['delta_psnr_rgb'] >= -.05 for r in summaries)
        temporal = all(r['temporal_ratio'] <= .9 for r in summaries)
        worst = min(r['delta_psnr_rgb'] for r in local)
        gates[name] = {'all_domain_seed_quality': quality, 'all_domain_seed_time': temporal,
                       'worst_condition_delta_psnr': worst, 'passed': quality and temporal and worst >= -.5}
    assert len(pairs) == 120 and len(effects) == 24
    frontier = []
    for domain, seed in domain_seeds:
        current = means[(domain, seed, 'response')]
        dominating = []
        for method in ALPHAS:
            reference = means[(domain, seed, method)]
            better_quality = reference['psnr_rgb'] >= current['psnr_rgb']
            better_time = reference['temporal_error'] <= current['temporal_error']
            strict = reference['psnr_rgb'] > current['psnr_rgb'] or reference['temporal_error'] < current['temporal_error']
            if better_quality and better_time and strict:
                dominating.append(method)
        frontier.append({'domain': domain, 'seed': seed, 'fixed_methods_dominating_response': dominating})
    independent = all(gates[n]['passed'] for n in ['response_vs_scalar', 'response_vs_rgb', 'response_vs_fixed05'])
    nondominated = not any(r['fixed_methods_dominating_response'] for r in frontier)
    quality_passed = independent and nondominated
    groups, domains = defaultdict(list), []
    for r in seeds:
        groups[(r['domain'], r['method'])].append(r)
    for (domain, method), values in sorted(groups.items()):
        assert {r['seed'] for r in values} == {0, 1}
        domains.append({'domain': domain, 'method': method,
                        'parameters': 4076 if method == 'scalar' else 4116 if method in ['rgb', 'response'] else 4075,
                        **{m: mean(r[m] for r in values) for m in METRICS}})
    groups, alpha = defaultdict(list), []
    for r in stats:
        groups[(r['domain'], r['train_seed'], r['method'])].append(r)
    for (domain, seed, method), values in sorted(groups.items()):
        count = sum(r['accepted_pixels'] for r in values)
        alpha.append({'domain': domain, 'seed': seed, 'method': method, 'accepted_pixels': count,
                      'pixel_weighted_mean_alpha': sum((r['mean_accepted_alpha'] or 0)*r['accepted_pixels'] for r in values)/count if count else None,
                      'histogram_20_bins_0_to_1': [sum(r['alpha_histogram_20_bins_0_to_1'][i] for r in values) for i in range(20)]})
    assert len(domains) == 18 and len(alpha) == 32
    out = BASE/'report/exploration-r45-response-recursion'
    out.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        (out/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    for name, value in [('domain_summary.json', domains), ('domain_seed_summary.json', seeds),
                        ('domain_seed_effects.json', effects), ('paired_by_condition.json', pairs),
                        ('fixed_frontier.json', frontier), ('alpha_domain_seed_summary.json', alpha)]:
        write(name, value)
    write('audit.json', {'passed': True, 'rows': 180, 'new_neural_sequences': 20,
          'independent_cue_screen_passed': independent, 'fixed_frontier_nondominated': nondominated,
          'registered_quality_passed': quality_passed, 'cost_confirmed': False,
          'freeze_local_family': not quality_passed, 'gates': gates, 'source_sha256': sources,
          'scope': 'Complete DEV only. Cue is a deterministic rescaling of feature energy, not new variance theory. No handset or final test.'})
    lines = ['# R45 complete fixed controller comparison', '',
             'Both seeds, three controls each4000 new updates, same frozen square-MSE base inherited8000. All alpha points retained; 180 quality rows, 20 new CNN sequences obtaining cues and replaying existing raw. DEV only.', '',
             '|Domain|Method|Parameters|PSNR|SSIM|Temporal↓|Edge↓|', '|---|---|---:|---:|---:|---:|---:|']
    for r in domains:
        lines.append('|'+r['domain']+'|'+r['method']+'|'+str(r['parameters'])+'|'
                     +'|'.join(f'{r[m]:.6f}' for m in METRICS)+'|')
    lines += ['', '|Comparison|All domain/seed quality|All domain/seed time|Worst ΔPSNR|Gate|',
              '|---|---|---|---:|---|']
    for name, g in gates.items():
        lines.append(f"|{name}|{g['all_domain_seed_quality']}|{g['all_domain_seed_time']}|{g['worst_condition_delta_psnr']:.6f}|{g['passed']}|")
    lines += ['', f'Independent cue screen: {independent}; fixed-alpha frontier nondominated: {nondominated}; registered full quality screen: {quality_passed}. Cost not measured.', '',
              'The response arm must pass scalar/RGB/fixed.5 comparisons jointly. Generic learned fusion or better constant alpha is not cue evidence. All stateful outputs retain own-raw motion and RGB history. Internal response cue is rescaled feature RMS, not calibrated uncertainty or an independent derivative invention.']
    (out/'tables.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'rows': 180, 'independent_cue': independent, 'frontier_nondominated': nondominated,
                      'registered_quality': quality_passed}), flush=True)


if __name__ == '__main__':
    main()
