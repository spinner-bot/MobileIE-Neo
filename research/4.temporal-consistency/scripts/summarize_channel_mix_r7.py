"""Audit train-only proxy calibration and actual static/video frozen controls."""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

from exploration_r2_data import ROOT
from explore_temporal_controls import save_json, sha256
from feature_mix_r7 import SPECS

BASE = ROOT/'research/4.temporal-consistency/results'
METRICS = ['psnr_rgb', 'ssim_rgb', 'temporal_error', 'edge_mae']


def jsonl(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def mean(rows, keys):
    return {key: float(np.mean([r[key] for r in rows if r[key] is not None])) for key in keys}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    channel, static = BASE/'exploration-r7-channel', BASE/'exploration-r7b-static'
    static_rows = jsonl(static/'per_condition.jsonl')
    if len(static_rows) != 432 or json.loads((static/'runtime.json').read_text())['rows'] != 432:
        raise ValueError('static budget incomplete')
    raw_static = {(r['scene'], r['crop'], r['photons'], r['jpeg']): r for r in static_rows if r['method'] == 'pretrained'}
    static_delta = []
    for r in static_rows:
        ref = raw_static[(r['scene'], r['crop'], r['photons'], r['jpeg'])]
        static_delta.append({k: r[k] for k in ['scene', 'crop', 'photons', 'jpeg', 'method']} | {
            'psnr_delta': r['mean_draw_psnr']-ref['mean_draw_psnr'],
            'variance_relative_delta': r['variance_normalized']/ref['variance_normalized']-1,
            'noise_sd_relative_delta': r['noise_sd_255']/ref['noise_sd_255']-1,
            'adjacent_mae_relative_delta': r['adjacent_static_mae_255']/ref['adjacent_static_mae_255']-1})
    grouped = collections.defaultdict(list)
    for r in static_rows:
        grouped[r['method']].append(r)
    static_summary = []
    for method, rr in grouped.items():
        dd = [r for r in static_delta if r['method'] == method]
        static_summary.append({'method': method, **mean(rr, ['mean_draw_psnr','noise_sd_255','adjacent_static_mae_255']),
            **mean(dd, ['psnr_delta','variance_relative_delta','noise_sd_relative_delta','adjacent_mae_relative_delta']),
            'positive_psnr_conditions': sum(r['psnr_delta'] > 0 for r in dd),
            'lower_variance_conditions': sum(r['variance_relative_delta'] < 0 for r in dd),
            'both_improved': sum(r['psnr_delta'] > 0 and r['variance_relative_delta'] < 0 for r in dd)})
    save_json(out/'static_summary.json', static_summary)
    save_json(out/'static_paired_conditions.json', static_delta)
    rows, frames, source_files, event_records = [], [], {}, []
    max_first = 0.
    for part, count in [('pre', 760), ('post', 608)]:
        folder = BASE/('exploration-r7b-video-'+part)
        rr = jsonl(folder/'per_sequence.jsonl')
        if len(rr) != count or json.loads((folder/'runtime.json').read_text())['rows'] != count:
            raise ValueError('video budget incomplete')
        rows += rr
        frames += jsonl(folder/'per_frame.jsonl')
        events = json.loads((folder/'input_events.json').read_text())
        event_records.append(events)
        checks = json.loads((folder/'first_frame_checks.json').read_text())
        max_first = max(max_first, max(r['normalized_first_frame_max_abs_difference'] for r in checks))
        for path in folder.iterdir():
            if path.is_file():
                source_files[str(path.relative_to(ROOT))] = sha256(path)
    if event_records[0] != event_records[1]:
        raise ValueError('different input event decisions')
    if len({(r['method'],r['scene'],r['input_reset'],r['fusion']) for r in rows}) != 1368:
        raise ValueError('duplicate video rows')
    grouped = collections.defaultdict(list)
    for r in rows:
        grouped[(r['domain'],r['method'],r['input_reset'],r['fusion'])].append(r)
    domain_summary = [{'domain': d, 'method': m, 'input_reset': reset, 'fusion': fusion,
        'conditions': len(rr), **mean(rr,METRICS)} for (d,m,reset,fusion),rr in sorted(grouped.items())]
    save_json(out/'domain_summary.json', domain_summary)
    raw_reference = {(r['scene'],r['input_reset'],r['fusion']): r for r in rows if r['method'] == 'pretrained'}
    deltas = []
    for r in rows:
        ref = raw_reference[(r['scene'],r['input_reset'],r['fusion'])]
        deltas.append({k:r[k] for k in ['domain','scene','method','input_reset','fusion']} | {k:r[k]-ref[k] for k in METRICS})
    save_json(out/'video_paired_conditions.json', deltas)
    g1 = {(r['scene'],r['input_reset'],r['fusion']):r for r in rows if r['method']=='body_pre_selected_g1'}
    g1_difference = max(abs(r[k]-g1[(r['scene'],r['input_reset'],r['fusion'])][k])
        for r in rows if r['method']=='body_post_selected_g1' for k in METRICS)
    if g1_difference > 1e-8:
        raise ValueError('full replacement unexpectedly depends on square placement')
    event_frames = [r for r in frames if r['domain'].startswith('event_') and r['frame']==12]
    save_json(out/'event_frame12.json', event_frames)
    save_json(out/'input_event_audit.json', event_records[0])
    for entry in json.loads((channel/'cache_index.json').read_text())['conditions']:
        if sha256(ROOT/entry['path']) != entry['sha256']:
            raise ValueError('train-source pixels changed')
    repeat = json.loads((static/'batch_repeat_audit.json').read_text())
    save_json(out/'audit.json', {'train_source_static_conditions':48,'video_conditions':38,'static_rows':432,'video_rows':1368,
        'selection':json.loads((channel/'selection.json').read_text()),
        'first_frame_max_abs_normalized_difference':max_first, 'g1_pre_post_metric_max_abs':g1_difference,
        'static_batch1_batch8_psnr_max_abs_delta':max(abs(r['batch1_minus_batch8']['mean_draw_psnr']) for r in repeat),
        'max_static_state_storage_bytes_128':max(r['state_storage_bytes_128'] for r in static_rows),
        'video_source_files':source_files,
        'scope':'train-source calibration and reused exploratory video; covariance proxy is not a moving-video guarantee'})
    lines=['# Frozen channel/placement and input-reset controls','',
           'Untrained official weights; reused exploratory inputs. Low temporal error is better. Static proxy uses train images only.','']
    for domain in ['test','legacy_synthetic','event_light_step','event_cut']:
        lines += ['## '+domain,'','| Method | Reset | Fusion | PSNR | SSIM | Temporal |','|---|---|---|---:|---:|---:|']
        for r in domain_summary:
            if r['domain']==domain:
                lines.append(f"| {r['method']} | {r['input_reset']} | {r['fusion']} | {r['psnr_rgb']:.5f} | {r['ssim_rgb']:.5f} | {r['temporal_error']:.5f} |")
        lines.append('')
    (out/'tables.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
