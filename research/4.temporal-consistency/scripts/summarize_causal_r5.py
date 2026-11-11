"""Audit complete fixed-budget R5 evidence; summarize paired domain/clip effects."""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

from exploration_r2_data import ROOT
from explore_temporal_controls import save_json, sha256
from prepare_cache_r5 import CACHE_INDEX

BASE = ROOT/'research/4.temporal-consistency'
KEYS = ['psnr_rgb', 'ssim_rgb', 'edge_mae', 'temporal_error']


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def means(rows):
    return {key: float(np.mean([r[key] for r in rows if r[key] is not None])) for key in KEYS}


def method(variant, history='causal', post='raw'):
    return f'{variant}__{history}__{post}'


def cluster(scene, domain):
    if domain.startswith('event_'):
        return scene.split('_')[0]
    if domain == 'legacy_synthetic':
        return scene.split('_rho')[0]
    return scene


def paired(rows, first, second, domain):
    def collect(name):
        grouped = collections.defaultdict(list)
        for r in rows:
            if r['domain'] == domain and r['method'] == name:
                grouped[r['scene']].append(r)
        return {k: means(v) for k, v in grouped.items()}
    a, b = collect(first), collect(second)
    if a.keys() != b.keys():
        raise ValueError('unmatched sequence scope')
    per_scene = {s: {k: a[s][k]-b[s][k] for k in KEYS} for s in a}
    clusters = collections.defaultdict(list)
    for scene, delta in per_scene.items():
        clusters[cluster(scene, domain)].append(delta)
    grouped = [means(v) for v in clusters.values()]
    rng = np.random.default_rng(501)
    indices = rng.integers(0, len(grouped), (2000, len(grouped)))
    summary = {}
    for k in KEYS:
        values = np.array([r[k] for r in grouped])
        boot = values[indices].mean(1)
        summary[k] = {'mean': float(values.mean()), 'median': float(np.median(values)),
            'positive_clusters': int((values > 0).sum()), 'clusters': len(values),
            'exploratory_cluster_bootstrap_95': np.percentile(boot, [2.5, 97.5]).tolist()}
    return {'first_minus_second': [first, second], 'domain': domain,
            'by_input_condition': per_scene, 'summary': summary,
            'scope': 'train seeds and input-noise repeats averaged within scene; legacy has only 2 source clips; exploratory intervals are not confirmatory inference'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--include-followups', action='store_true')
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    rows, inputs, frame_paths = [], {}, []
    specs = [('exploration-r5-eval-'+part, expected, '') for part, expected in [('spatial', 456), ('temporal', 912), ('shift', 608)]]
    if args.include_followups:
        specs += [('exploration-r5b-eval-'+part, expected, '_st') for part, expected in [('plain', 152), ('temporal', 456), ('shift', 304)]]
        specs += [('exploration-r5c-eval-spatial-post', 152, '')]
    def alias(r, suffix):
        if not suffix:
            return r
        variant = r['variant']+suffix
        return {**r, 'variant': variant, 'method': method(variant, r['history'], r['post'])}
    for part, expected, suffix in specs:
        folder = BASE/'results'/part
        seq_path = folder/'per_sequence.jsonl'
        values = jsonl(seq_path)
        if len(values) != expected or json.loads((folder/'runtime.json').read_text())['rows'] != expected:
            raise ValueError('incomplete evaluation '+part)
        if len({(r['method'], r['scene'], r['train_seed']) for r in values}) != expected:
            raise ValueError('duplicate evaluations')
        for checkpoint in json.loads((folder/'checkpoints.json').read_text()):
            if sha256(ROOT/checkpoint['path']) != checkpoint['sha256']:
                raise ValueError('checkpoint hash mismatch')
        rows.extend(alias(r, suffix) for r in values)
        inputs[str(seq_path.relative_to(ROOT))] = sha256(seq_path)
        frame_paths.append((folder/'per_frame.jsonl', suffix))
    baseline = BASE/'results/exploration-r5-cache/baseline_per_sequence.json'
    for r in json.loads(baseline.read_text()):
        rows.append({**r, 'variant': 'pretrained', 'train_seed': None, 'history': 'causal',
                     'post': r['method'].split('__')[1], 'method': method('pretrained', post=r['method'].split('__')[1])})
    inputs[str(baseline.relative_to(ROOT))] = sha256(baseline)
    groups = collections.defaultdict(list)
    by_scene = collections.defaultdict(list)
    for r in rows:
        groups[(r['domain'], r['method'])].append(r)
        by_scene[(r['domain'], r['method'], r['scene'])].append(r)
    domain_rows = [{'domain': d, 'method': m, 'input_conditions': len({r['scene'] for r in values}),
                    'evaluations': len(values), **means(values)} for (d, m), values in sorted(groups.items())]
    save_json(out/'domain_summary.json', domain_rows)
    save_json(out/'by_scene.json', [{'domain': d, 'method': m, 'scene': s, **means(v)} for (d, m, s), v in sorted(by_scene.items())])
    contrasts = [('temporal_bilinear', 'causal', 'spatial_bilinear', 'causal'),
                 ('temporal_bilinear', 'causal', 'temporal_bilinear', 'current'),
                 ('temporal_bilinear', 'causal', 'temporal_bilinear', 'off'),
                 ('temporal_post', 'causal', 'temporal_post', 'current'),
                 ('temporal_post', 'causal', 'temporal_post', 'off'),
                 ('tsm_learned', 'causal', 'tsm_learned', 'current'),
                 ('tsm_fixed', 'causal', 'tsm_fixed', 'current'),
                 ('plain', 'causal', 'r2_task', 'causal')]
    contrasts += [(v, 'causal', 'plain', 'causal') for v in ['spatial_bilinear', 'temporal_bilinear', 'temporal_post', 'tsm_learned', 'tsm_fixed']]
    if args.include_followups:
        contrasts += [('temporal_post', 'causal', 'spatial_post', 'causal'),
                      ('spatial_post', 'causal', 'plain', 'causal')]
        contrasts += [(v+'_st', 'causal', v, 'causal') for v in ['plain', 'temporal_bilinear', 'tsm_learned']]
        contrasts += [(v+'_st', 'causal', 'plain_st', 'causal') for v in ['temporal_bilinear', 'tsm_learned']]
        contrasts += [(v+'_st', 'causal', v+'_st', 'current') for v in ['temporal_bilinear', 'tsm_learned']]
    deltas = [paired(rows, method(a, ah, post), method(b, bh, post), domain)
              for a, ah, b, bh in contrasts for post in ['raw', 'half01']
              for domain in ['test', 'legacy_synthetic', 'event_light_step', 'event_cut']]
    save_json(out/'paired_deltas.json', deltas)
    training, repeat, traces = [], [], {}
    training_specs = ['exploration-r5-'+part for part in ['spatial', 'temporal', 'shift']]
    if args.include_followups:
        training_specs += ['exploration-r5b-'+part for part in ['plain', 'temporal', 'shift']]
        training_specs += ['exploration-r5c-spatial-post']
    for part in training_specs:
        for folder in sorted((BASE/'results'/part).iterdir()):
            trace = jsonl(folder/'training.jsonl')
            if [r['step'] for r in trace] != list(range(1, 1001)):
                raise ValueError('incomplete training trace')
            config = json.loads((folder/'config.json').read_text())
            seed = config['seed']
            identities = [r['sample_identity'] for r in trace]
            if seed in traces and traces[seed] != identities:
                raise ValueError('unmatched training data stream')
            traces[seed] = identities
            for path in sorted(folder.glob('checkpoint-step*.json')):
                item = json.loads(path.read_text())
                if sha256(ROOT/item['path']) != item['sha256']:
                    raise ValueError('saved checkpoint changed')
            training.append({'scope': part, 'run': folder.name, 'steps': len(trace), 'initial_task': trace[0]['task'],
                'last100_task': float(np.mean([r['task'] for r in trace[-100:]])),
                'final_gamma': trace[-1].get('gamma'), 'final_branch_l2': trace[-1].get('branch_l2')})
            if (folder/'r4b-repeat-audit.json').exists():
                repeat.append({'run': folder.name, **json.loads((folder/'r4b-repeat-audit.json').read_text())})
    expected_runs = 20 if args.include_followups else 12
    if len(training) != expected_runs or len(repeat) != 2 or not all(r['exact_match'] for r in repeat):
        raise ValueError('training/repeat budget mismatch')
    save_json(out/'training_audit.json', {'runs': training, 'total_updates': expected_runs*1000, 'same_streams_all_modes': True, 'r4b_repeat': repeat})
    event_rows = []
    for path, suffix in frame_paths:
        groups = collections.defaultdict(list)
        for original in jsonl(path):
            r = alias(original, suffix)
            if r['domain'].startswith('event_'):
                groups[(r['domain'], r['scene'], r['method'], r['train_seed'])].append(r)
        for (domain, scene, name, seed), frames in groups.items():
            frames.sort(key=lambda r: r['frame'])
            if len(frames) != 24 or (domain == 'event_cut' and frames[12]['temporal_error'] is not None):
                raise ValueError('event timing or cut mask wrong')
            bright = lambda rr: float(np.mean([abs(r['output_mean']-r['target_mean']) for r in rr]))
            row = {'domain': domain, 'scene': scene, 'method': name, 'train_seed': seed,
                'psnr_frame12': frames[12]['psnr_rgb'], 'ssim_frame12': frames[12]['ssim_rgb'],
                'psnr_first3': float(np.mean([r['psnr_rgb'] for r in frames[12:15]])),
                'psnr_before3': float(np.mean([r['psnr_rgb'] for r in frames[9:12]])),
                'psnr_late8': float(np.mean([r['psnr_rgb'] for r in frames[16:]])),
                'brightness_error_frame12': bright(frames[12:13]),
                'brightness_error_first3': bright(frames[12:15]), 'brightness_error_late8': bright(frames[16:])}
            if domain == 'event_light_step':
                jump = frames[12]['target_mean']-frames[11]['target_mean']
                row['brightness_response_gain'] = (frames[12]['output_mean']-frames[11]['output_mean'])/jump
            event_rows.append(row)
    save_json(out/'event_diagnostics.json', event_rows)
    save_json(out/'audit.json', {'model_sequence_evaluations': 3040 if args.include_followups else 1976, 'teacher_sequence_evaluations': 76,
        'input_conditions': 38, 'training_runs': expected_runs, 'training_updates': expected_runs*1000,
        'cache_index_sha256': sha256(CACHE_INDEX), 'inputs': inputs,
        'frame_files': {str(path.relative_to(ROOT)): sha256(path) for path, _ in frame_paths},
        'scope': 'reused exploratory data, 6 real ND clips incl one R3 overlap, 2 legacy source clips; no locked test claims'})
    lines = ['# R5 complete domain means', '', 'All train seeds/input repeats averaged; lower temporal/edge is better. Exploratory, no test selection.', '']
    for domain in ['test', 'legacy_synthetic', 'event_light_step', 'event_cut']:
        lines += ['## '+domain, '', '| Method | PSNR | SSIM | temporal error | edge MAE |', '|---|---:|---:|---:|---:|']
        for r in domain_rows:
            if r['domain'] == domain:
                lines.append(f"| {r['method']} | {r['psnr_rgb']:.5f} | {r['ssim_rgb']:.5f} | {r['temporal_error']:.5f} | {r['edge_mae']:.5f} |")
        lines.append('')
    (out/'tables.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
