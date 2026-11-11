"""Audit empirical latent-moment closures and group all fixed conditions without RGB claims."""
import collections
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from correlated_observations_r19 import encode
from exploration_r2_data import ROOT
from explore_temporal_controls import save_json, sha256
from probe_latent_moments import array_hash

BASE = ROOT/'research/4.temporal-consistency'


def main():
    folder = BASE/'results/exploration-r29-latent-moments'
    out = BASE/'report/exploration-r29'
    out.mkdir(parents=True, exist_ok=False)
    runtime = json.loads((folder/'runtime.json').read_text())
    assert runtime['rows'] == 160 and runtime['current_feature_replays'] == 80
    assert runtime['clean_node_cache_entries'] == 40 and not runtime['cuda_initialized']
    rows = [json.loads(line) for line in (folder/'moments.jsonl').read_text().splitlines()]
    keys = ['scene', 'photons', 'rho', 'jpeg', 'weight', 'stage']
    assert len(rows) == len({tuple(r[k] for k in keys) for r in rows}) == 160
    max_closure = max(v for r in rows for v in r['closure_scaled'].values())
    assert max_closure <= 1e-10
    replays = json.loads((folder/'current_replays.json').read_text())
    assert len(replays) == 80 and all(r['current_feature_sha256_exact'] for r in replays)
    # Independently verify that the noiseless reference input does not change with photons/rho.
    conditions = json.loads((BASE/'results/exploration-r19/cache_index.json').read_text())['conditions']
    target_hashes, clean_hashes = {}, {}
    for r in conditions:
        path = ROOT/r['path']
        assert sha256(path) == r['sha256']
        with np.load(path, allow_pickle=False) as data:
            target = data['target']
            target_sha = array_hash(target)
            if r['scene'] in target_hashes:
                assert target_hashes[r['scene']] == target_sha
            target_hashes[r['scene']] = target_sha
            clean = encode((target.astype(np.float64)**2.2*.35)[None], r['jpeg'])
            key = (r['scene'], r['jpeg'])
            digest = array_hash(clean)
            if key in clean_hashes:
                assert clean_hashes[key] == digest
            clean_hashes[key] = digest
    assert len(target_hashes) == 2 and len(clean_hashes) == 4
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r['objective'], r['stage'], r['photons'], r['rho'], r['jpeg'])].append(r)
    summary = []
    for (objective, stage, photons, rho, jpeg), values in sorted(groups.items()):
        mean = lambda fn: float(np.mean([fn(r) for r in values]))
        record = {'objective': objective, 'stage': stage, 'photons': photons, 'rho': rho, 'jpeg': jpeg, 'rows': len(values),
                  'feature_empirical_correlation': mean(lambda r: r['feature_empirical_correlation']),
                  'coefficient_a': mean(lambda r: r['coefficient_a'])}
        for name in ['self_mean_shift_term', 'self_variance_term', 'cross_mean_shift_term', 'cross_covariance_term']:
            record[name] = {measure: mean(lambda r, name=name, measure=measure: r[name][measure]) for measure in ['signed_mean', 'mean_abs', 'rms']}
            record[name]['rms_over_reference_q_rms'] = mean(lambda r, name=name: r[name]['rms']/max(r['reference_q_rms'], 1e-30))
        record['latent_bias'] = {name: {'rms': mean(lambda r, name=name: r['empirical_latent_bias'][name]['rms']),
                                      'rms_over_reference_q_rms': mean(lambda r, name=name: r['empirical_latent_bias'][name]['rms']/max(r['reference_q_rms'], 1e-30))}
                                 for name in ['self', 'postmean', 'pre_square', 'cross']}
        summary.append(record)
    save_json(out/'group_summary.json', summary)
    sources = {str(p.relative_to(ROOT)): sha256(p) for p in folder.iterdir() if p.is_file()}
    save_json(out/'audit.json', {'rows': 160, 'current_feature_replays_exact': 80, 'clean_node_cache_entries': 40,
              'reference_input_consistency_groups': 4, 'target_consistency_groups': 2, 'max_scaled_closure': max_closure,
              'source_sha256': sources, 'training_updates': 0, 'cuda_initialized': False,
              'scope': 'Two TRAIN images, existing repeated synthetic noise, latent reference is noiseless encoded LOW input. No RGB quality or unbiased-network claim.'})
    lines = ['# Frozen latent conditional moments', '', 'RMS terms do not add; averages are empirical, not population-unbiasedness proofs.', '',
             '| Objective | Node | Photons | Rho | JPEG | Mean-shift RMS | Variance RMS | Self bias RMS | Cross bias RMS | Feature corr |',
             '|---|---|---:|---:|---|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"|{r['objective']}|{r['stage']}|{r['photons']}|{r['rho']}|{r['jpeg']}|{r['self_mean_shift_term']['rms']:.8g}|{r['self_variance_term']['rms']:.8g}|{r['latent_bias']['self']['rms']:.8g}|{r['latent_bias']['cross']['rms']:.8g}|{r['feature_empirical_correlation']:.6f}|")
    (out/'tables.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
