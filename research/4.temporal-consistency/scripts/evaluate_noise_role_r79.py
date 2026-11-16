"""All final R79 causal/current controls, common CPU spatial and temporal metrics."""
import argparse
import json
import os
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import cv2
import numpy as np
import torch

from noise_role_io_r79 import ROOT, BASE, PROCESS, digest, write, append, stamp, verify, owned_cache
from noise_role_model_r79 import MODES, models, precision
from causal_reference_evaluation_r75 import metas
from exploration_r2_data import evaluate
from prepare_cache_r5 import load_one

ROUTES = (('spatial', 'current'), ('balanced', 'causal'), ('balanced', 'current'),
          ('curriculum', 'causal'), ('curriculum', 'current'))


def final_markers():
    verify()
    markers = {}
    for seed in (0, 1):
        path = BASE/f'results/exploration-r79-stage001-seed{seed}/final.json'
        marker = json.loads(path.read_text())
        terminal = json.loads((ROOT/marker['runtime']).read_text())
        assert marker['completed_normally'] and marker['final_quality_eligible']
        assert marker['completed_steps'] == marker['next_data_step'] == marker['registered_steps'] == 4000
        assert terminal['completed_normally'] and terminal['stop_reason'] == 'complete'
        assert terminal['completed_steps'] == 4000 and max(marker['final_reload'].values()) <= 1e-6
        assert marker['freeze_sha256'] == digest(PROCESS/'source-freeze001.json')
        checkpoint = ROOT/marker['path']
        assert checkpoint.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
        assert digest(checkpoint) == marker['sha256']
        markers[seed] = marker
    return markers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    cv2.setNumThreads(1); precision()
    markers = final_markers(); scenes, sources = metas()
    saved = torch.load(ROOT/markers[args.seed]['path'], map_location='cpu', weights_only=True)
    assert saved['experiment'] == 'R79' and saved['completed_steps'] == 4000 and saved['seed'] == args.seed
    nets = models(args.seed)
    for name, net in nets.items():
        net.load_state_dict(saved['models'][name], strict=True)
        net.eval().requires_grad_(False)
        assert all(torch.isfinite(p).all() for p in net.parameters())
    del saved
    out = BASE/f'results/exploration-r79-eval-seed{args.seed}'
    out.mkdir(exist_ok=False)
    cache = owned_cache(f'experiments/exploration-r79/eval001/seed{args.seed}')
    write(out/'config.json', dict(at=stamp(), seed=args.seed, checkpoints=markers, sources=sources,
        expected_rows=50, expected_frame_rows=2240, expected_CNN_centers=2240, routes=ROUTES,
        scope='Repeated paired DEV, full spatial shape and raw previous frame, final4000 only. Not final test or phone.'))
    began, complete, rows, frames, cnn = time.perf_counter(), False, 0, 0, 0
    try:
        for meta in scenes:
            data = load_one(meta)
            for name, route in ROUTES:
                net = nets[name]; output = np.empty(data['gt'].shape, dtype=np.float32)
                started = time.perf_counter()
                previous = None
                with torch.inference_mode():
                    for t, raw in enumerate(data['low']):
                        current = torch.from_numpy(raw.transpose(2, 0, 1).copy()[None].astype(np.float32))/255
                        past = current if previous is None or route == 'current' else previous
                        actual = net.transform(current, past)
                        assert torch.isfinite(actual).all()
                        output[t] = actual[0].clamp(0, 1).permute(1, 2, 0).contiguous().numpy()*255
                        previous = current
                        cnn += 1
                identity = dict(seed=args.seed, model=name, history=route, label=f'{name}_{route}',
                    domain=meta['domain'], scene=meta['scene'], source_video=meta['source_video'], method='raw',
                    parameters=sum(p.numel() for p in net.parameters()), updates=4000)
                append(out/'stage_cost.jsonl', dict(**identity, seconds=time.perf_counter()-started,
                    frames=len(output), previous_RGB_bytes=0 if name == 'spatial' or route == 'current' else previous.numel()*4,
                    scope='Shared CPU descriptive time only; no handset or warmed fair throughput claim.'))
                path = cache/f'{meta["scene"]}--{name}--{route}.npy'
                np.save(path, output, allow_pickle=False)
                append(out/'prediction_manifest.jsonl', dict(**identity, path=str(path.relative_to(ROOT)),
                    sha256=digest(path), shape=list(output.shape), checkpoint_sha256=markers[args.seed]['sha256']))
                metrics, details = evaluate(output, data['gt'], data['reference_flow'], data['reference_mask'])
                append(out/'per_sequence.jsonl', dict(**identity, **metrics))
                for row in details:
                    append(out/'per_frame.jsonl', dict(**identity, **row))
                rows += 1; frames += len(details)
                print(json.dumps(dict(seed=args.seed, scene=meta['scene'], model=name, history=route,
                                      completed_rows=rows)), flush=True)
                del output
            del data
        assert (rows, frames, cnn) == (50, 2240, 2240)
        assert not torch.cuda.is_initialized()
        complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(out/'runtime.json', dict(ended=stamp(), completed_normally=complete, seed=args.seed,
            quality_rows=rows, frame_rows=frames, new_CNN_centers=cnn, new_SGD=0,
            CUDA_initialized=torch.cuda.is_initialized(), seconds=time.perf_counter()-began))


if __name__ == '__main__':
    main()
