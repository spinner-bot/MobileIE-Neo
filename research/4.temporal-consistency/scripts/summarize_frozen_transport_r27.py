"""Audit frozen native operator controls, exact current aliases and common CPU replay."""
import collections
import json

from evaluate_aggregation_r20 import metric_delta
from exploration_r2_data import ROOT
from explore_temporal_controls import save_json, sha256
from summarize_causal_r5 import KEYS, jsonl, means, paired
from transport_fst_r27 import MODES

BASE = ROOT/'research/4.temporal-consistency'
ID = ['domain', 'scene', 'weight', 'objective', 'train_seed', 'mode', 'history', 'fusion']


def label(r):
    return '|'.join(str(r[k]) for k in ID)


def method(r):
    return '__'.join([r['weight'], r['mode'], r['history'], r['fusion']])


def main():
    folder = BASE/'results/exploration-r27-frozen-cpu'
    out = BASE/'report/exploration-r27-frozen'
    out.mkdir(parents=True, exist_ok=False)
    runtime = json.loads((folder/'runtime.json').read_text())
    assert tuple(runtime[k] for k in ['rows', 'actual_inference_sequences', 'metric_computations', 'metric_aliases']) == (240, 200, 140, 100)
    rows = jsonl(folder/'per_sequence.jsonl')
    assert len(rows) == len({label(r) for r in rows}) == 240
    replay_checks = []
    sources = {}
    for filename in ['per_sequence.jsonl', 'per_frame.jsonl']:
        frame = filename == 'per_frame.jsonl'
        values = jsonl(folder/filename)
        seen = {}
        for r in values:
            key = (label(r), r['frame']) if frame else label(r)
            if r['metric_alias_of'] is not None:
                other = seen[(r['metric_alias_of'], r['frame']) if frame else r['metric_alias_of']]
                assert all(r[k] == other[k] for k in [*KEYS, 'flicker'])
                if frame:
                    assert all(r[k] == other[k] for k in ['output_mean', 'target_mean'])
            assert key not in seen
            seen[key] = r
        reference_path = BASE/'results/exploration-r21-reference-plain-cpu'/filename
        reference = {}
        for r in jsonl(reference_path):
            if r['domain'] == 'dev_native' and r['step'] == 4000 and not r['input_reset'] and r['fusion'] == 'raw':
                reference[(r['scene'], r['train_seed'], *((r['frame'],) if frame else ()))] = r
        maximum = 0.
        count = 0
        for r in values:
            if r['objective'] == 'charb' and r['mode'] == 'copy' and r['history'] == 'causal' and r['fusion'] == 'raw':
                key = (r['scene'], r['train_seed'], *((r['frame'],) if frame else ()))
                maximum = max(maximum, metric_delta(r, reference[key]))
                count += 1
        assert count == len(reference) and maximum <= 1e-10
        replay_checks.append({'file': filename, 'rows': count, 'max_abs_metric': maximum, 'reference': str(reference_path.relative_to(ROOT))})
        sources[str(reference_path.relative_to(ROOT))] = sha256(reference_path)
    first = json.loads((folder/'first_frame_audit.json').read_text())
    limits = json.loads((folder/'current_limit_audit.json').read_text())
    assert len(first) == len(limits) == 100
    assert max(r['max_abs_normalized'] for r in first+limits) <= 2e-5 and all(r['all_frames_exact'] for r in limits)
    for r in json.loads((folder/'checkpoints.json').read_text()):
        assert sha256(ROOT/r['path']) == r['sha256']
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r['weight'], r['objective'], r['mode'], r['history'], r['fusion'])].append(r)
    summary = [{'weight': w, 'objective': o, 'mode': m, 'history': h, 'fusion': f, 'clips': len(rr), **means(rr)}
               for (w, o, m, h, f), rr in sorted(groups.items())]
    save_json(out/'domain_summary.json', summary)
    contrasts = []
    values = [{**r, 'method': method(r)} for r in rows]
    for weight in sorted({r['weight'] for r in rows}):
        for mode in MODES[1:]:
            for output in ['raw', 'postwarp01', 'postwarp05']:
                contrasts.append(paired(values, '__'.join([weight, mode, 'causal', 'raw']), '__'.join([weight, 'copy', 'causal', output]), 'dev_native'))
    save_json(out/'paired_deltas.json', contrasts)
    for path in folder.iterdir():
        if path.is_file():
            sources[str(path.relative_to(ROOT))] = sha256(path)
    save_json(out/'audit.json', {'rows': 240, 'metric_computations': 140, 'metric_aliases': 100, 'actual_inference_sequences': 200,
              'common_CPU_replay': replay_checks, 'sequence_and_frame_alias_exact': True,
              'current_limits_all_frames_exact': True, 'first_frame_max_abs_normalized': max(r['max_abs_normalized'] for r in first),
              'source_sha256': sources, 'training_updates': 0,
              'scope': 'Frozen forward intervention on four exploratory native DEV clips. Not trained-method quality, convergence, unseen-test or novelty evidence.'})
    lines = ['# Frozen body-FST operator controls (native DEV, common CPU)', '', '| Weights | Mode | History | Output | PSNR | SSIM | Temporal | Edge |',
             '|---|---|---|---|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"|{r['weight']}|{r['mode']}|{r['history']}|{r['fusion']}|{r['psnr_rgb']:.6f}|{r['ssim_rgb']:.6f}|{r['temporal_error']:.6f}|{r['edge_mae']:.6f}|")
    (out/'tables.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
