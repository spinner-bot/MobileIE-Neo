"""Final-only full paired DEV; exogenous wrong history generated after true sequences."""
import argparse
import json
import os
import time
import traceback

import cv2
import numpy as np
import torch

from self_reference_io_r82 import ROOT, BASE, digest, write, append, stamp, verify, owned_cache, precision, final_markers
from self_reference_r82 import make_models
from causal_reference_evaluation_r75 import metas, reference_predictions
from evaluate_cvvdp_references_r66 import cached
from prepare_cache_r5 import load_one
from exploration_r2_data import evaluate
from render_dense_history_r64 import recursive


def donor_scene(meta):
    if meta['domain'] == 'davis_train_pilot':
        name = dict(bear='lucia', lucia='walking', walking='bear')[meta['source_video']]
        return f'{name}_photons{meta["photons"]}'
    return dict(video_15='video_47', video_47='video_91', video_91='video_121', video_121='video_15')[meta['scene']]


def tensor(frame):
    return torch.from_numpy(np.asarray(frame).transpose(2, 0, 1).copy()[None].astype(np.float32))/255


def image(value):
    assert bool(torch.isfinite(value).all())
    return value[0].clamp(0, 1).permute(1, 2, 0).contiguous().numpy()*255


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    precision(); cv2.setNumThreads(1); frozen = verify()
    markers = final_markers(); scenes, indices = metas()
    references = {(r['seed'], r['scene'], r['label']): r for r in reference_predictions()}
    saved = torch.load(ROOT/markers[args.seed]['path'], map_location='cpu', weights_only=True)
    assert saved['experiment'] == 'R82' and saved['seed'] == args.seed and saved['completed_steps'] == 8000
    nets = make_models(args.seed)
    for name, net in nets.items():
        net.load_state_dict(saved['models'][name], strict=True); net.eval().requires_grad_(False)
        assert all(bool(torch.isfinite(p).all()) for p in net.parameters())
    del saved
    out = BASE/f'results/exploration-r82-eval-seed{args.seed}'; out.mkdir(exist_ok=False)
    cache = owned_cache(f'experiments/exploration-r82/eval001/seed{args.seed}')
    write(out/'config.json', dict(at=stamp(), pid=os.getpid(), seed=args.seed, frozen=frozen,
        final_checkpoint=markers[args.seed], cache_indices=indices,
        wrong_history_donors={m['scene']: donor_scene(m) for m in scenes},
        expected_new_quality_rows=50, expected_frame_rows=2240, expected_H_calls=3504,
        scope='Repeated DEV; original frozen raw cache reuse avoids experiment F0 calls, not deployment F0 cost.'))
    began, complete, count, frames = time.perf_counter(), False, 0, 0
    entries = {}

    def record(meta, label, values, data, extra=None):
        nonlocal count, frames
        identity = dict(seed=args.seed, domain=meta['domain'], scene=meta['scene'], source_video=meta['source_video'],
            label=label, parameters=4379, new_parameters=304, frozen_backbone_updates=8000, head_updates=8000,
            deployable=label != 'centered_wrong')
        path = cache/f'{meta["scene"]}--{label}.npy'; np.save(path, values, allow_pickle=False)
        entry = dict(**identity, path=str(path.relative_to(ROOT)), sha256=digest(path), shape=list(values.shape),
            checkpoint_sha256=markers[args.seed]['sha256'], **(extra or {}))
        entries[meta['scene'], label] = entry; append(out/'prediction_manifest.jsonl', entry)
        metrics, details = evaluate(values, data['gt'], data['reference_flow'], data['reference_mask'])
        append(out/'per_sequence.jsonl', dict(**identity, **metrics))
        for row in details:
            append(out/'per_frame.jsonl', dict(**identity, **row))
        count += 1; frames += len(details)
        print(json.dumps(dict(seed=args.seed, scene=meta['scene'], label=label, completed_rows=count)), flush=True)

    try:
        for meta in scenes:
            data = load_one(meta)
            ref = references[args.seed, meta['scene'], 'MobileIE_raw']
            assert ref['checkpoint_sha256'] == markers[args.seed]['backbone']['checkpoint_sha256']
            raw = cached(ref); assert raw.shape == data['gt'].shape
            for name, net in nets.items():
                net.reset(); values = np.empty(raw.shape, dtype=np.float32)
                started = time.perf_counter()
                with torch.inference_mode():
                    for t, frame in enumerate(raw):
                        values[t] = image(net.step(tensor(frame)))
                append(out/'stage-cost.jsonl', dict(seed=args.seed, scene=meta['scene'], label=name,
                    seconds=time.perf_counter()-started, frames=len(values), state_bytes=net.state_bytes(),
                    excludes_frozen_backbone=True, scope='Shared CPU descriptive timing, not complete deployment or phone.'))
                record(meta, name, values, data, dict(frozen_input=ref))
                if name == 'spatial':
                    started = time.perf_counter(); recursive_values = recursive(values)
                    append(out/'stage-cost.jsonl', dict(seed=args.seed, scene=meta['scene'], label='spatial_recursive_self05',
                        seconds=time.perf_counter()-started, frames=len(values),
                        scope='Post stage only; own raw half-FB, not evaluation GT flow or complete deployment.'))
                    record(meta, 'spatial_recursive_self05', recursive_values, data,
                        dict(input_prediction=entries[meta['scene'], 'spatial'], flow='Own raw half-FB recursive .5'))
                    del recursive_values
                del values
            maximum = 0.
            with torch.inference_mode():
                for t, frame in enumerate(raw):
                    current = tensor(frame)
                    same = nets['centered'](current, None if t == 0 else current, history='current')
                    assert bool(torch.isfinite(same).all())
                    maximum = max(maximum, float((same-current).abs().max()))
            assert maximum <= 1e-6
            append(out/'self-alias.jsonl', dict(seed=args.seed, scene=meta['scene'],
                actual_computed_frames=len(raw), normalized_max_delta=maximum, alias=ref, new_quality_rows=0))
            del raw, data
        # All true centered predictions already exist; inject the fixed other source's t-1.
        for meta in scenes:
            data = load_one(meta); ref = references[args.seed, meta['scene'], 'MobileIE_raw']; raw = cached(ref)
            donor = entries[donor_scene(meta), 'centered']; past = cached(donor)
            assert past.shape == raw.shape, 'No resize or alternate donor fallback permitted.'
            values = np.empty(raw.shape, dtype=np.float32)
            with torch.inference_mode():
                for t, frame in enumerate(raw):
                    values[t] = image(nets['centered'](tensor(frame), None if t == 0 else tensor(past[t-1])))
            record(meta, 'centered_wrong', values, data, dict(frozen_input=ref,
                injected_other_true_causal_output=donor, exogenous_teacher_forced=True))
            del raw, data, past, values
        calls = {name: net.head_calls for name, net in nets.items()}
        assert (count, frames, sum(calls.values())) == (50, 2240, 3504)
        assert not torch.cuda.is_initialized(); verify(); complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(ended=stamp(), pid=os.getpid(), completed_normally=complete, seed=args.seed,
            quality_rows=count, frame_rows=frames, H_calls={name:net.head_calls for name,net in nets.items()},
            new_frozen_backbone_calls=0, new_SGD=0, CUDA_initialized=torch.cuda.is_initialized(),
            seconds=time.perf_counter()-began))


if __name__ == '__main__':
    main()
