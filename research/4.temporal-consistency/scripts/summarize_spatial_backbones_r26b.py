"""Audit same-weight spatial deployment and contrast actual history with strong post controls."""
import collections
import json

from evaluate_aggregation_r20 import metric_delta
from evaluate_spatial_backbones_r26b import keys
from exploration_r2_data import ROOT
from explore_temporal_controls import save_json, sha256
from summarize_causal_r5 import KEYS, jsonl, means, paired
from summarize_loss_controls_r26 import DOMAINS, ID, label, method

BASE = ROOT/'research/4.temporal-consistency'


def main():
    out = BASE/'report/exploration-r26b'
    out.mkdir(parents=True, exist_ok=False)
    rows, actual, replays, runtimes, first, sources = [], [], [], [], [], {}
    for architecture in ['pre_fixedhalf', 'post_fixedhalf']:
        folder = BASE/'results'/('exploration-r26b-eval-'+architecture)
        original = BASE/'results'/('exploration-r26-eval-'+architecture)
        run = json.loads((folder/'runtime.json').read_text())
        assert (run['rows'], run['actual_inference_sequences'], run['metric_computations'], run['metric_aliases']) == (2944, 368, 2048, 896)
        assert run['device'] == 'cuda:0'
        runtimes.append(run)
        cfg = json.loads((folder/'config.json').read_text())
        assert cfg['device'] == 'cuda:0' and cfg['steps'] == [1000, 4000]
        markers = json.loads((folder/'checkpoints.json').read_text())
        old_markers = json.loads((original/'checkpoints.json').read_text())
        assert len(markers) == 8
        for marker, old in zip(markers, old_markers):
            assert marker['path'] == old['path'] and marker['sha256'] == old['sha256'] == sha256(ROOT/marker['path'])
        events = json.loads((folder/'input_events.json').read_text())
        assert events == json.loads((original/'input_events.json').read_text()) and len(events) == 46
        event_map = {(r['domain'], r['scene']): bool(r['reset_frames']) for r in events}
        for filename in ['per_sequence.jsonl', 'per_frame.jsonl']:
            frame = filename == 'per_frame.jsonl'
            def rowkey(r):
                return (label(r), r['frame']) if frame else label(r)
            old_rows = jsonl(original/filename)
            reference = {rowkey(r): r for r in old_rows if r['history'] == 'current'}
            values = jsonl(folder/filename)
            seen, maximum, count = {}, 0., 0
            for r in values:
                assert r['architecture'] == architecture and r['history'] == 'current'
                _, key = keys('plain', 'current', r['input_reset'], event_map[(r['domain'], r['scene'])], r['fusion'])
                assert (r['effective_model_reset'], r['effective_output_reset']) == (key[0], key[2])
                k = rowkey(r)
                assert k not in seen
                if r['metric_alias_of'] is not None:
                    alias = (r['metric_alias_of'], r['frame']) if frame else r['metric_alias_of']
                    other = seen[alias]
                    assert all(r[q] == other[q] for q in [*KEYS, 'flicker'])
                    if frame:
                        assert all(r[q] == other[q] for q in ['output_mean', 'target_mean'])
                if r['fusion'] in ['raw', 'half01']:
                    maximum = max(maximum, metric_delta(r, reference[k]))
                    count += 1
                seen[k] = r
            assert maximum <= 1e-10
            assert count == len(reference)
            replays.append({'architecture': architecture, 'file': filename, 'rows': count, 'max_abs_metric': maximum})
            if not frame:
                rows.extend(values)
                actual.extend(r for r in old_rows if r['history'] == 'causal')
            for path in [folder/filename, original/filename]:
                sources[str(path.relative_to(ROOT))] = sha256(path)
        first.extend(json.loads((folder/'first_frame_audit.json').read_text()))
        for path in folder.iterdir():
            if path.is_file():
                sources[str(path.relative_to(ROOT))] = sha256(path)
    assert len(rows) == len({label(r) for r in rows}) == 5888
    assert max(r['max_abs_pixel'] for r in first) <= 1e-4
    groups = collections.defaultdict(list)
    scenes = collections.defaultdict(list)
    for r in rows:
        key = tuple(r[k] for k in ['domain', 'architecture', 'objective', 'step', 'input_reset', 'fusion'])
        groups[key].append(r)
        scenes[(r['scene'], *key)].append(r)
    summary = [{'domain': d, 'architecture': a, 'objective': o, 'step': s, 'input_reset': reset, 'fusion': f, **means(rr)}
               for (d, a, o, s, reset, f), rr in sorted(groups.items())]
    save_json(out/'domain_summary.json', summary)
    save_json(out/'by_scene.json', [{'scene': c, 'domain': d, 'architecture': a, 'objective': o, 'step': s, 'input_reset': reset, 'fusion': f, **means(rr)}
             for (c, d, a, o, s, reset, f), rr in sorted(scenes.items())])
    contrasts = []
    for reset in [False, True]:
        values = [{**r, 'method': method(r['architecture'], r['objective'], r['step'], r['history'], r['fusion'])}
                  for r in rows+actual if r['input_reset'] == reset]
        for a in ['pre_fixedhalf', 'post_fixedhalf']:
            for o in ['charb', 'mse']:
                for step in [1000, 4000]:
                    for output in ['raw', 'half01']:
                        for baseline in ['raw', 'half01', 'postwarp01', 'postwarp05']:
                            for domain in DOMAINS:
                                contrasts.append({**paired(values, method(a, o, step, 'causal', output), method(a, o, step, 'current', baseline), domain),
                                                  'input_reset': reset, 'contrast': 'actual_history_minus_same_weight_spatial_post'})
    save_json(out/'paired_deltas.json', contrasts)
    save_json(out/'audit.json', {'rows': len(rows), 'actual_inference_sequences': 736, 'metric_computations': 4096, 'metric_aliases': 1792,
              'first_frame_max_abs_pixel': max(r['max_abs_pixel'] for r in first), 'historical_current_replays': replays,
              'common_device': 'cuda:0', 'source_sha256': sources,
              'scope': 'Same trained weights, stronger single-frame/classical-post controls. Exploratory repeated conditions, no new structure or mobile latency claim.'})
    lines = ['# Same-weight spatial mappings plus output controls', '', '| Domain | Architecture | Objective | Step | Reset | Output | PSNR | SSIM | Temporal | Edge |',
             '|---|---|---|---:|---|---|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"|{r['domain']}|{r['architecture']}|{r['objective']}|{r['step']}|{r['input_reset']}|{r['fusion']}|{r['psnr_rgb']:.6f}|{r['ssim_rgb']:.6f}|{r['temporal_error']:.6f}|{r['edge_mae']:.6f}|")
    (out/'tables.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
