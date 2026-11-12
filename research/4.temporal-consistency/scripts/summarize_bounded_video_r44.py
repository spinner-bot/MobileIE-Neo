"""Close all factorial arms and three separately registered scientific screens."""
from collections import defaultdict
import json
from statistics import mean

from bounded_video_r44 import MODELS
from dynamic_training_r34 import BASE, ROOT, digest

METRICS = ['psnr_rgb', 'ssim_rgb', 'temporal_error', 'edge_mae']
COMPARISONS = [
    ('activation_mse', 'bounded_mse', 'raw', 'square_mse', 'raw'),
    ('objective_square', 'square_temporal', 'raw', 'square_mse', 'raw'),
    ('objective_bounded', 'bounded_temporal', 'raw', 'bounded_mse', 'raw'),
    ('structure', 'bounded_temporal', 'raw', 'square_temporal', 'raw'),
    ('stateless_vs_mse_recursive', 'bounded_temporal', 'raw', 'square_mse', 'recursive_self05'),
    ('stateless_vs_temporal_recursive', 'bounded_temporal', 'raw', 'square_temporal', 'recursive_self05'),
    ('combined_vs_mse_recursive', 'bounded_temporal', 'recursive_self05', 'square_mse', 'recursive_self05'),
    ('combined_vs_temporal_recursive', 'bounded_temporal', 'recursive_self05', 'square_temporal', 'recursive_self05')]


def main():
    rows, sources, kappas = [], {}, []
    for seed in [0, 1]:
        folder = BASE/f'results/exploration-r44-stage001-eval-seed{seed}'
        runtime = json.loads((folder/'runtime.json').read_text())
        assert runtime['completed_normally'] and runtime['rows'] == 120
        assert runtime['neural_sequences'] == 40 and not runtime['cuda_initialized']
        values = list(map(json.loads, (folder/'per_sequence.jsonl').read_text().splitlines()))
        assert len(values) == 120 and {r['train_seed'] for r in values} == {seed}
        rows += values
        training = BASE/f'results/exploration-r44-stage001-seed{seed}'
        end = [json.loads(v) for v in (training/'training.jsonl').read_text().splitlines()[-4:]]
        assert {r['model'] for r in end} == set(MODELS) and {r['new_step'] for r in end} == {4000}
        kappas += [{'seed': seed, 'model': r['model'], 'kappa': r['kappa']} for r in end]
        for path in [folder/'config.json', folder/'runtime.json', folder/'per_sequence.jsonl', training/'training.jsonl']:
            sources[str(path.relative_to(ROOT))] = digest(path)
    assert len(rows) == 240
    lookup = {(r['domain'], r['scene'], r['train_seed'], r['model'], r['method']): r for r in rows}
    assert len(lookup) == 240
    groups, seeds, domains = defaultdict(list), [], []
    for row in rows:
        groups[(row['domain'], row['model'], row['train_seed'], row['method'])].append(row)
    for (domain, model, seed, method), values in sorted(groups.items()):
        assert len(values) == (6 if domain == 'davis_train_pilot' else 4)
        seeds.append({'domain': domain, 'model': model, 'seed': seed, 'method': method,
                      **{m: mean(r[m] for r in values) for m in METRICS}})
    assert len(seeds) == 48
    means = {(r['domain'], r['model'], r['seed'], r['method']): r for r in seeds}
    pairs, effects, gates = [], [], {}
    domain_names = ['davis_train_pilot', 'paired_real_ND_DEV']
    conditions = sorted({(r['domain'], r['scene'], r['train_seed']) for r in rows})
    assert len(conditions) == 20
    for name, model, method, ref_model, ref_method in COMPARISONS:
        chosen = []
        for domain, scene, seed in conditions:
            current, ref = [lookup[(domain, scene, seed, m, f)] for m, f in [(model, method), (ref_model, ref_method)]]
            assert ref['temporal_error'] > 0
            pair = {'comparison': name, 'domain': domain, 'scene': scene, 'seed': seed,
                    **{'delta_'+m: current[m]-ref[m] for m in METRICS},
                    'temporal_ratio': current['temporal_error']/ref['temporal_error']}
            pairs.append(pair)
            chosen.append(pair)
        selected = []
        for domain in domain_names:
            for seed in [0, 1]:
                current, ref = [means[(domain, m, seed, f)] for m, f in [(model, method), (ref_model, ref_method)]]
                effect = {'comparison': name, 'domain': domain, 'seed': seed,
                          **{'delta_'+m: current[m]-ref[m] for m in METRICS},
                          'temporal_ratio': current['temporal_error']/ref['temporal_error']}
                effects.append(effect)
                selected.append(effect)
        quality = all(r['delta_psnr_rgb'] >= -.05 for r in selected)
        temporal = all(r['temporal_ratio'] <= .90 for r in selected)
        worst = min(r['delta_psnr_rgb'] for r in chosen)
        gates[name] = {'all_domain_seed_quality': quality, 'all_domain_seed_time': temporal,
                       'worst_condition_delta_psnr': worst, 'passed': quality and temporal and worst >= -.5}
    assert len(pairs) == 160 and len(effects) == 32
    groups = defaultdict(list)
    for row in seeds:
        groups[(row['domain'], row['model'], row['method'])].append(row)
    for (domain, model, method), values in sorted(groups.items()):
        assert {r['seed'] for r in values} == {0, 1}
        domains.append({'domain': domain, 'model': model, 'method': method,
                        'deployment_parameters': 4077 if model.startswith('bounded') else 4075,
                        **{m: mean(r[m] for r in values) for m in METRICS}})
    interactions = []
    for domain in domain_names:
        for seed in [0, 1]:
            sm, st, bm, bt = [means[(domain, m, seed, 'raw')] for m in MODELS]
            interactions.append({'domain': domain, 'seed': seed,
                                 **{m: bt[m]-bm[m]-st[m]+sm[m] for m in METRICS},
                                 'scope': 'Descriptive 2x2 metric difference, not statistical significance.'})
    structure = gates['structure']['passed']
    stateless = all(gates[n]['passed'] for n in ['stateless_vs_mse_recursive', 'stateless_vs_temporal_recursive'])
    combination = all(gates[n]['passed'] for n in ['combined_vs_mse_recursive', 'combined_vs_temporal_recursive'])
    out = BASE/'report/exploration-r44-bounded-video'
    out.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        (out/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    for name, value in [('domain_summary.json', domains), ('domain_seed_summary.json', seeds),
                        ('paired_by_condition.json', pairs), ('domain_seed_effects.json', effects),
                        ('factorial_interactions.json', interactions), ('final_kappa.json', kappas)]:
        write(name, value)
    write('audit.json', {'passed': True, 'new_rows': 240, 'new_neural_sequences': 80,
          'registered_structure_passed': structure, 'registered_stateless_passed': stateless,
          'registered_combination_quality_passed': combination, 'combination_cost_confirmed': False,
          'freeze_local_family': not (structure or stateless or combination), 'gates': gates,
          'source_sha256': sources,
          'scope': 'Fixed DEV factorial, inherited4000 + new4000 and fresh Adam. Known activation/regularization; no phone or independent test.'})
    lines = ['# R44 fixed factorial: complete final evaluation', '',
             'All four models and both seeds, inherited R41 4000 plus new4000 each, fresh Adam. 240 new quality rows / 80 CNN sequences. Both domains and every condition are retained; no seed/checkpoint selection.', '',
             '|Domain|Model|Output|Parameters|PSNR|SSIM|Temporal↓|Edge↓|', '|---|---|---|---:|---:|---:|---:|---:|']
    for row in domains:
        lines.append('|'+row['domain']+'|'+row['model']+'|'+row['method']+'|'+str(row['deployment_parameters'])+'|'
                     +'|'.join(f'{row[m]:.6f}' for m in METRICS)+'|')
    lines += ['', '|Comparison|All domain/seed quality|All domain/seed time|Worst condition ΔPSNR|Joint screen|',
              '|---|---|---|---:|---|']
    for name, gate in gates.items():
        lines.append(f"|{name}|{gate['all_domain_seed_quality']}|{gate['all_domain_seed_time']}|{gate['worst_condition_delta_psnr']:.6f}|{gate['passed']}|")
    lines += ['', f'Registered structure: {structure}; stateless deployment: {stateless}; combination quality: {combination}. Combination cost has not been measured; it cannot be reported as the stateless result.', '',
              'Activation/individual-objective comparisons are descriptive controls. Registered gates require every domain/seed PSNR≥−.05dB, temporal ratio≤.90 and all20 condition PSNR deltas≥−.5dB. Rational operations remain in inference; finite/recursive use explicit own-raw motion/history. GT motion is training-only and estimated, not motion truth. No final-test, phone cost or novelty claim follows from these tables.']
    (out/'tables.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'rows': 240, 'structure': structure, 'stateless': stateless, 'combination_quality': combination}), flush=True)


if __name__ == '__main__':
    main()
