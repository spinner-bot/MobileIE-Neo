"""Audit and summarize local moments, independent oracle controls and RGB covariance."""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

from exploration_r2_data import ROOT
from explore_temporal_controls import save_json, sha256

BASE = ROOT/'research/4.temporal-consistency/results'


def jsonl(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def stats(values):
    a = np.array(values, dtype=float)
    return {'mean': float(a.mean()), 'median': float(np.median(a)), 'min': float(a.min()), 'max': float(a.max())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    original, centered, components = [BASE/name for name in ['exploration-r6-moments', 'exploration-r6b-centered', 'exploration-r6c-rgb-components']]
    features = jsonl(original/'feature_per_channel.jsonl')
    rgb = jsonl(original/'rgb_per_condition.jsonl')+jsonl(centered/'rgb_per_condition.jsonl')
    comp = json.loads((components/'per_condition.json').read_text())
    layer = jsonl(centered/'layer_per_condition.jsonl')
    references = jsonl(original/'oracle_references.jsonl')
    if [len(features), len(rgb), len(comp), len(layer), len(references)] != [5184, 1008, 144, 144, 216]:
        raise ValueError('incomplete fixed budgets')
    cfg = json.loads((original/'config.json').read_text())
    for row in cfg['models'].values():
        if sha256(ROOT/row['path']) != row['sha256']:
            raise ValueError('checkpoint changed')
    cache = json.loads((original/'cache_index.json').read_text())['conditions']
    for row in cache:
        if sha256(ROOT/row['path']) != row['sha256']:
            raise ValueError('cache changed')
    groups = collections.defaultdict(list)
    for r in features:
        groups[(r['scene'], r['photons'], r['jpeg'], r['model'], r['layer'], r['draw_subset'])].append(r)
    feature_conditions = []
    for (scene, photons, jpeg, model, location, subset), rr in groups.items():
        total = sum(r['total_square_variance'] for r in rr)
        item = {'scene': scene, 'photons': photons, 'jpeg': jpeg, 'model': model, 'layer': location, 'draw_subset': subset,
                'total_square_variance': total}
        for name in ['linear', 'quadratic', 'cross']:
            item[name+'_ratio'] = sum(r[name+'_energy'] for r in rr)/total
        feature_conditions.append(item)
    save_json(out/'feature_per_condition.json', feature_conditions)
    grouped = collections.defaultdict(list)
    for r in feature_conditions:
        if r['draw_subset'] == 'all64':
            grouped[(r['model'], r['layer'])].append(r)
    save_json(out/'feature_summary.json', [{'model': m, 'layer': l,
        **{name+'_ratio': stats([r[name+'_ratio'] for r in rr]) for name in ['linear', 'quadratic', 'cross']}}
        for (m, l), rr in grouped.items()])
    raw = {(r['scene'], r['photons'], r['jpeg'], r['model'], r['clipped']): r for r in rgb if r['intervention'] == 'raw'}
    grouped = collections.defaultdict(list)
    paired = []
    for r in rgb:
        grouped[(r['model'], r['intervention'], r['clipped'])].append(r)
        if r['intervention'] != 'raw':
            ref = raw[(r['scene'], r['photons'], r['jpeg'], r['model'], r['clipped'])]
            paired.append({k: r[k] for k in ['scene', 'photons', 'jpeg', 'model', 'intervention', 'clipped']} | {
                'psnr_delta': r['mean_draw_psnr']-ref['mean_draw_psnr'],
                'noise_sd_relative_delta': r['noise_sd_255']/ref['noise_sd_255']-1,
                'variance_relative_delta': r['variance_normalized']/ref['variance_normalized']-1,
                'bias_mse_delta': r['bias_mse_normalized']-ref['bias_mse_normalized']})
    rgb_summary = [{'model': m, 'intervention': i, 'clipped': c, 'conditions': len(rr),
        **{k: float(np.mean([r[k] for r in rr])) for k in ['mean_draw_psnr', 'noise_sd_255', 'mse_normalized', 'bias_mse_normalized', 'variance_normalized']}}
        for (m, i, c), rr in sorted(grouped.items())]
    save_json(out/'rgb_summary.json', rgb_summary)
    save_json(out/'paired_conditions.json', paired)
    groups = collections.defaultdict(list)
    for r in paired:
        groups[(r['model'], r['intervention'], r['clipped'])].append(r)
    paired_summary = []
    for (m, i, c), rr in sorted(groups.items()):
        per_image = collections.defaultdict(list)
        for r in rr:
            per_image[r['scene']].append(r)
        paired_summary.append({'model': m, 'intervention': i, 'clipped': c, 'conditions': len(rr),
            'psnr_delta': stats([r['psnr_delta'] for r in rr]),
            'noise_sd_relative_delta': stats([r['noise_sd_relative_delta'] for r in rr]),
            'positive_psnr_conditions': sum(r['psnr_delta'] > 0 for r in rr),
            'lower_noise_conditions': sum(r['noise_sd_relative_delta'] < 0 for r in rr),
            'both_improved_conditions': sum(r['psnr_delta'] > 0 and r['noise_sd_relative_delta'] < 0 for r in rr),
            'by_image': {s: {key: float(np.mean([r[key] for r in values])) for key in ['psnr_delta', 'noise_sd_relative_delta']}
                         for s, values in per_image.items()}})
    save_json(out/'paired_summary.json', paired_summary)
    component_summary = []
    for model in cfg['models']:
        for clipped in [False, True]:
            rr = [r for r in comp if r['model'] == model and r['clipped'] == clipped]
            names = rr[0]['components']
            fine_names = rr[0]['split_interaction']['components']
            qi = fine_names.index('feature_quadratic_mean_gate')
            dqi = fine_names.index('gate_quadratic_interaction')
            q_marginal, q_variance = [], []
            for r in rr:
                detail = r['split_interaction']; cov = np.array(detail['covariance'])
                q_marginal.append(detail['covariance_with_total_over_total_variance'][qi]+detail['covariance_with_total_over_total_variance'][dqi])
                q_variance.append((cov[qi, qi]+cov[dqi, dqi]+2*cov[qi, dqi])/detail['actual_rgb_variance'])
            component_summary.append({'model': model, 'clipped': clipped, 'conditions': len(rr),
                'marginal_covariance_fraction': {name: stats([r['covariance_with_total_over_total_variance'][i] for r in rr]) for i, name in enumerate(names)},
                'energy_fraction': {name: stats([r['component_variance_over_total_variance'][i] for r in rr]) for i, name in enumerate(names)},
                'combined_quadratic_marginal_fraction': stats(q_marginal),
                'combined_quadratic_energy_fraction': stats(q_variance),
                'scope': 'algebraic covariance, may be negative, does not predict intervention benefit; body-linear includes upstream nonlinear effects'})
    save_json(out/'component_summary.json', component_summary)
    save_json(out/'centered_local_summary.json', [{'model': m, 'layer': l,
        'local_variance_ratio': stats([r['local_variance_ratio'] for r in layer if r['model'] == m and r['layer'] == l]),
        'calibration_mean_shift_over_original_variance': stats([r['mean_shift_mse_over_original_variance'] for r in layer if r['model'] == m and r['layer'] == l])}
        for m in cfg['models'] for l in ['head', 'body']])
    repeat_max = {}
    for folder in [centered, components]:
        audit = json.loads((folder/'raw_repeat_audit.json').read_text())
        repeat_max[folder.name] = max(v for r in audit for v in r['max_abs_by_metric'].values())
    save_json(out/'audit.json', {'static_conditions': 24, 'source_images': 6, 'draw_images': 3072,
        'feature_channel_rows': 5184, 'rgb_rows': 1008, 'covariance_rows': 144, 'oracle_reference_rows': 216,
        'raw_repeat_max_abs': repeat_max,
        'feature_identity_max_relative_error': max(r['formula_max_abs_over_max_direct'] for r in features),
        'rgb_mse_closure_max_abs': max(r['closure_abs'] for r in rgb),
        'covariance_closure_max_abs': max(r['variance_closure_abs'] for r in comp),
        'covariance_reconstruction_max_abs': max(r['centered_reconstruction_max_abs'] for r in comp),
        'rounding_variance_max_ratio': max(r['component_variance_over_total_variance'][r['components'].index('rounding_residual')] for r in comp if not r['clipped']),
        'inputs': {str(path.relative_to(ROOT)): sha256(path) for folder in [original, centered, components] for path in folder.iterdir() if path.is_file()},
        'scope': 'static repeated synthetic diagnosis, not real-video performance or independent test'})
    lines = ['# R6/R6b/R6c fixed-draw mechanism diagnosis', '', 'Clipped RGB; means over 24 conditions, not 24 independent scenes. Oracle controls are not deployable.', '',
             '| Model | Intervention | Mean PSNR | Noise SD (0–255 levels) |', '|---|---|---:|---:|']
    for r in rgb_summary:
        if r['clipped']:
            lines.append(f"| {r['model']} | {r['intervention']} | {r['mean_draw_psnr']:.5f} | {r['noise_sd_255']:.5f} |")
    lines += ['', '## Exact RGB decomposition', '', 'Cov(component, output)/Var(output); correlated quantities, may be negative.', '',
              '| Model | Body linear / mean gate | Body quadratic / mean gate | Gate / mean feature | Gate-feature interaction | Combined quadratic incl interaction |',
              '|---|---:|---:|---:|---:|---:|']
    for r in component_summary:
        if r['clipped']:
            d = r['marginal_covariance_fraction']
            vals = [d[k]['mean'] for k in ['feature_linear_mean_gate','feature_quadratic_mean_gate','gate_mean_feature','gate_feature_interaction']]
            vals.append(r['combined_quadratic_marginal_fraction']['mean'])
            lines.append('| '+r['model']+' | '+' | '.join(f'{v:.6f}' for v in vals)+' |')
    (out/'tables.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
