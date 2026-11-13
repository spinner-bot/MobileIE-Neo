"""Final R74 motion inference on frozen spatial caches, both transport controls."""
import argparse
from datetime import datetime
import json
import time
import traceback

import cv2
import numpy as np
import torch

from motion_transport_r74 import BASE, ROOT, PROTOCOL, MODES, MotionTransport, RGBStream
from check_motion_transport_r74 import write
from dynamic_training_r34 import digest
from evaluate_cvvdp_references_r66 import lines, append, cached
from exploration_r2_data import evaluate
from prepare_cache_r5 import load_one


def metas():
    paths = [BASE/f'results/{n}/index.json' for n in ['exploration-r30-cache', 'exploration-r10-cache']]
    rows = json.loads(paths[0].read_text())['scenes']
    rows += [dict(r, domain='paired_real_ND_DEV', source_video=r['scene'])
             for r in json.loads(paths[1].read_text())['scenes'] if r['domain'] == 'native_crop']
    assert len(rows) == 10
    return rows, {str(p.relative_to(ROOT)): digest(p) for p in paths}


def reference_predictions():
    out = BASE/'process/exploration-r73/references001'
    assert json.loads((out/'runtime.json').read_text())['completed_normally']
    path = out/'prediction_manifest.jsonl'
    rows = [r for r in lines(path) if r['model'] == 'R44_plain']
    assert len(rows) == 40
    return rows, {str(path.relative_to(ROOT)): digest(path)}


def reference_quality():
    out = BASE/'report/exploration-r70-classical-joint'
    assert json.loads((out/'audit.json').read_text())['completed']
    path = out/'all_conditions.jsonl'
    rows = [dict(r, model='R44_plain', history='current', parameters=4075, spatial_updates=8000, motion_updates=0)
            for r in lines(path) if r['label'] in ['MobileIE_raw', 'MobileIE_recursive_self05']]
    assert len(rows) == 40 and all('JOD' in r for r in rows)
    return rows, {str(path.relative_to(ROOT)): digest(path)}


def key(r): return tuple(r[n] for n in ['seed', 'scene', 'model', 'history', 'method'])


def load_final(seed):
    # Both complete seeds are a prerequisite; no prefix or best-checkpoint evaluation.
    for other in [0, 1]:
        terminal = json.loads((BASE/f'results/exploration-r74-stage001-seed{other}/runtime.json').read_text())
        assert terminal['completed_normally'] and terminal['motion_model_updates'] == {n: 4000 for n in MODES}
    out = BASE/f'results/exploration-r74-stage001-seed{seed}'
    nets, markers = {}, {}
    for name in MODES:
        marker = json.loads((out/f'{name}-step4000.json').read_text())
        assert marker['new_step'] == 4000 and marker['spatial_updates'] == 8000
        assert marker['final_reload_passed'] and marker['final_reload_max_delta'] <= 1e-6
        path = ROOT/marker['path']
        assert path.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo') and digest(path) == marker['sha256']
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        assert checkpoint['model'] == name and checkpoint['seed'] == seed and checkpoint['new_step'] == 4000
        net = MotionTransport().eval().requires_grad_(False)
        net.load_state_dict(checkpoint['state_dict'], strict=True)
        assert sum(p.numel() for p in net.parameters()) == marker['motion_parameters'] == 21918
        assert all(torch.isfinite(p).all() for p in net.parameters())
        nets[name], markers[name] = net, marker
    return nets, markers


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--seed', type=int, choices=[0, 1], required=True)
    args = parser.parse_args(); torch.set_num_threads(2); cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True); assert not torch.cuda.is_initialized()
    nets, markers = load_final(args.seed); scenes, sources = metas()
    references, ref_sources = reference_predictions(); sources.update(ref_sources)
    raw_lookup = {r['scene']: r for r in references if r['seed'] == args.seed and r['method'] == 'raw'}
    assert len(raw_lookup) == 10
    out = BASE/f'results/exploration-r74-eval-seed{args.seed}'; out.mkdir(exist_ok=False)
    cache = ROOT/f'.cache/experiments/exploration-r74/eval001/seed{args.seed}'
    assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    cache.mkdir(parents=True, exist_ok=False)
    write(out/'config.json', dict(at=datetime.now().astimezone().isoformat(), seed=args.seed, checkpoints=markers,
          protocol_sha256=digest(PROTOCOL), sources=sources, expected_rows=40, expected_frames=1792,
          expected_motion_sequences=20, expected_nonfirst_motion_pairs=876, new_enhancement_CNN=0,
          first_frame_serialization='Exact cached raw; functional normalized-unit equality checked before avoiding a /255*255 round trip.',
          scope='Final motion4000/frozen spatial8000, reused paired DEV. Two routes share identical predicted fields. Not mobile/final.'))
    started, complete, count, frames, sequences, pairs = time.perf_counter(), False, 0, 0, 0, 0
    try:
        for meta in scenes:
            data = load_one(meta); entry = raw_lookup[meta['scene']]; raw = cached(entry)
            assert raw.shape == data['gt'].shape
            for name, net in nets.items():
                spatial_marker = json.loads((ROOT/markers[name]['spatial_source']['marker']).read_text())
                assert spatial_marker['sha256'] == entry['checkpoint_sha256'] == markers[name]['spatial_source']['checkpoint_sha256']
                identity = dict(seed=args.seed, domain=meta['domain'], scene=meta['scene'], source_video=meta['source_video'],
                      model=name, method='raw', parameters=25993, motion_parameters=21918, spatial_updates=8000, motion_updates=4000)
                streams = {'causal': RGBStream(net), 'zero_flow': RGBStream(net, zero_flow=True)}
                values = {route: [] for route in streams}; began = time.perf_counter()
                with torch.inference_mode():
                    for t in range(len(raw)):
                        current = torch.from_numpy(raw[t].transpose(2, 0, 1).copy()[None])/255
                        actual, field = streams['causal'].step(current)
                        control, _ = streams['zero_flow'].step(current, fields=field)
                        assert torch.isfinite(actual).all() and torch.isfinite(control).all()
                        if t:
                            pairs += 1
                        else:
                            assert torch.equal(actual, current) and torch.equal(control, current)
                        for route, value in [('causal', actual), ('zero_flow', control)]:
                            values[route].append(raw[0].copy() if t == 0 else
                                value[0].clamp(0, 1).permute(1, 2, 0).contiguous().numpy()*255)
                sequences += 1
                state_bytes = streams['causal'].state_storage_bytes()
                assert state_bytes == 6*raw.shape[1]*raw.shape[2]*4
                append(out/'stage_cost.jsonl', dict(**identity, seconds=time.perf_counter()-began, frames=len(raw),
                       motion_pairs=len(raw)-1, state_bytes_per_route=state_bytes,
                       scope='Descriptive shared-fields two-route evaluation; not a warmed fair cost benchmark or phone latency.'))
                for route, sequence in values.items():
                    output = np.stack(sequence); assert np.array_equal(output[0], raw[0]) and np.isfinite(output).all()
                    append(out/'first_frame.jsonl', dict(**identity, history=route, exact_cached_raw=True))
                    path = cache/f'{meta["scene"]}--{name}--{route}.npy'; np.save(path, output, allow_pickle=False)
                    append(out/'prediction_manifest.jsonl', dict(**identity, history=route, path=str(path.relative_to(ROOT)),
                           shape=list(output.shape), sha256=digest(path), raw_source=entry, checkpoint_sha256=markers[name]['sha256']))
                    metrics, detail = evaluate(output, data['gt'], data['reference_flow'], data['reference_mask'])
                    append(out/'per_sequence.jsonl', dict(**identity, history=route, **metrics))
                    for row in detail: append(out/'per_frame.jsonl', dict(**identity, history=route, **row))
                    count += 1; frames += len(detail)
                print(json.dumps(dict(**identity, completed_quality_rows=count)), flush=True)
            del data, raw
        assert (count, frames, sequences, pairs) == (40, 1792, 20, 876); complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(completed_normally=complete, quality_rows=count, frame_rows=frames,
              motion_sequences=sequences, nonfirst_motion_pairs=pairs, new_enhancement_CNN=0, new_SGD=0,
              CUDA_initialized=torch.cuda.is_initialized(), seconds=time.perf_counter()-started,
              ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__': main()
