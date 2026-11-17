"""Final-only shared-stem causal video quality; true and zero have separate states."""
import argparse
import json
import os
import time
import traceback

import cv2
import numpy as np
import torch

from shared_geometry_io_r88 import ROOT, BASE, PROCESS, verify, precision, final_markers, owned_cache, write, append, stamp, digest
from shared_geometry_r88 import make_heads, stem, CausalCore
from motion_transport_r74 import load_spatial
from causal_reference_evaluation_r75 import metas, reference_predictions
from evaluate_cvvdp_references_r66 import cached
from prepare_cache_r5 import load_one
from exploration_r2_data import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    precision(); cv2.setNumThreads(1)
    frozen = verify(json.loads((PROCESS/'evaluation-source-freeze001.json').read_text()))
    markers = final_markers(); scenes, indices = metas()
    refs = {(r['seed'], r['scene'], r['label']): r for r in reference_predictions()}
    backbone, source = load_spatial(args.seed)
    assert source == markers[args.seed]['backbone']
    saved = torch.load(ROOT/markers[args.seed]['path'], map_location='cpu', weights_only=True)
    assert saved['experiment'] == 'R88' and saved['seed'] == args.seed and saved['completed_steps'] == 2000
    heads, rank = make_heads(backbone, args.seed)
    for name, head in heads.items():
        head.load_state_dict(saved['models'][name], strict=True)
        heads[name] = head.folded().requires_grad_(False)
        assert all(bool(torch.isfinite(p).all()) for p in heads[name].parameters())
    del saved
    out = BASE/f'results/exploration-r88-eval-seed{args.seed}'; out.mkdir(exist_ok=False)
    cache = owned_cache(f'experiments/exploration-r88/eval001/seed{args.seed}')
    write(out/'config.json', dict(at=stamp(), pid=os.getpid(), seed=args.seed, frozen=frozen,
        final_checkpoint=markers[args.seed], cache_indices=indices, rank=rank,
        expected_new_quality_rows=40, expected_frame_rows=1792, expected_stem_frames=448,
        expected_descriptor_frames=896, expected_bidirectional_pairs=876,
        scope='Repeated DEV; reused original RGB, actual new shared stems; not full deployment cost or phone.'))
    began, complete, count, frames, stem_frames, desc_frames, pairs = time.perf_counter(), False, 0, 0, 0, 0, 0
    try:
        for meta in scenes:
            data = load_one(meta)
            ref = refs[args.seed, meta['scene'], 'MobileIE_raw']
            assert ref['checkpoint_sha256'] == source['checkpoint_sha256']
            raw = cached(ref); assert raw.shape == data['gt'].shape
            states = {name: CausalCore(('true', 'zero')) for name in heads}
            values = {name+'_'+route: np.empty_like(raw, dtype=np.float32)
                      for name in heads for route in ('true', 'zero')}
            started = time.perf_counter()
            with torch.inference_mode():
                for t, low in enumerate(data['input']):
                    tensor = torch.from_numpy(low.transpose(2, 0, 1).copy()[None].astype(np.float32))/255
                    features = stem(backbone, tensor); stem_frames += 1
                    for name, head in heads.items():
                        descriptor = head(features); desc_frames += 1
                        outputs = states[name].step(raw[t], descriptor)
                        for route, prediction in outputs.items():
                            assert np.isfinite(prediction).all()
                            values[name+'_'+route][t] = prediction
            elapsed = time.perf_counter()-started
            pairs += sum(state.pairs for state in states.values())
            append(out/'stage-cost.jsonl', dict(seed=args.seed, scene=meta['scene'], seconds=elapsed,
                frames=len(raw), pipelines=4, shared_stem=True, excludes_complete_F0=True,
                evaluation_state_bytes={name: state.state_bytes() for name, state in states.items()},
                deployment_single_route_state_bytes=3*raw.shape[1]*raw.shape[2]*4+
                    4*((raw.shape[1]+3)//4)*((raw.shape[2]+3)//4)*4,
                scope='Four-route evaluation stage descriptive timing; not candidate deployment or phone.'))
            for label, value in values.items():
                assert np.array_equal(value[0], raw[0])
                identity = dict(seed=args.seed, domain=meta['domain'], scene=meta['scene'],
                    source_video=meta['source_video'], label=label, parameters=4507, new_parameters=432,
                    frozen_backbone_updates=8000, head_updates=2000, deployable=True)
                path = cache/f'{meta["scene"]}--{label}.npy'; np.save(path, value, allow_pickle=False)
                append(out/'prediction_manifest.jsonl', dict(**identity, path=str(path.relative_to(ROOT)),
                    sha256=digest(path), shape=list(value.shape), checkpoint_sha256=markers[args.seed]['sha256'],
                    frozen_input=ref, descriptor_degrees=432 if label.startswith('plain') else 4*(108-rank),
                    zero_mask='Identical true accepted mask; separate own RGB history' if label.endswith('zero') else None))
                metrics, detail = evaluate(value, data['gt'], data['reference_flow'], data['reference_mask'])
                append(out/'per_sequence.jsonl', dict(**identity, **metrics))
                for row in detail:
                    append(out/'per_frame.jsonl', dict(**identity, **row))
                count += 1; frames += len(detail)
                print(json.dumps(dict(seed=args.seed, scene=meta['scene'], label=label, completed_rows=count)), flush=True)
            del data, raw, values, states
        assert (count, frames, stem_frames, desc_frames, pairs) == (40, 1792, 448, 896, 876)
        assert not torch.cuda.is_initialized(); verify(frozen); complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(ended=stamp(), pid=os.getpid(), completed_normally=complete, seed=args.seed,
            quality_rows=count, frame_rows=frames, stem_frames=stem_frames, descriptor_frames=desc_frames,
            bidirectional_pairs=pairs, new_full_F0=0, new_SGD=0, CUDA_initialized=torch.cuda.is_initialized(),
            seconds=time.perf_counter()-began))


if __name__ == '__main__':
    main()
