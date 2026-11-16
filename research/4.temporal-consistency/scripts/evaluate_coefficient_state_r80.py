"""Complete CPU readout experiment, prediction-before-update and independent GT diagnostic."""
import argparse
import json
import os
import time
import traceback

import cv2
import numpy as np
import torch

from coefficient_io_r80 import ROOT, BASE, PROCESS, LABELS, digest, write, append, stamp, verify, owned_cache
from coefficient_state_r80 import ConditionalReadout, PrefixRidge, PrefixTeacher, rgb_basis, rgb_predict
from motion_transport_r74 import load_spatial
from causal_reference_evaluation_r75 import metas, reference_predictions
from evaluate_cvvdp_references_r66 import cached
from prepare_cache_r5 import load_one
from exploration_r2_data import evaluate


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--seed', type=int, choices=(0, 1), required=True)
    args = parser.parse_args(); assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(2); cv2.setNumThreads(1); torch.use_deterministic_algorithms(True)
    frozen = verify(); model, source = load_spatial(args.seed); reader = ConditionalReadout(model)
    scenes, indices = metas(); references = reference_predictions()
    index = {(r['seed'], r['scene'], r['label']): r for r in references}
    out = BASE/f'results/exploration-r80-eval-seed{args.seed}'; out.mkdir(exist_ok=False)
    cache = owned_cache(f'experiments/exploration-r80/eval001/seed{args.seed}')
    write(out/'config.json', dict(at=stamp(), pid=os.getpid(), seed=args.seed, frozen=frozen,
        backbone=source, cache_indices=indices, expected_new_quality_rows=80, expected_frame_rows=3184,
        expected_new_predictions=30, expected_shared_core_frames=448, expected_ridge_updates=240,
        cached_raw_normalized_max_delta=2e-5, scope='All fixed paired DEV, full and tail. GT only explicit nondeploy diagnostic.'))
    began, complete, rows, frames, core, fits = time.perf_counter(), False, 0, 0, 0, 0
    try:
        for meta in scenes:
            data = load_one(meta)
            raw_entry = index[args.seed, meta['scene'], 'MobileIE_raw']
            rec_entry = index[args.seed, meta['scene'], 'MobileIE_recursive_self05']
            assert raw_entry['checkpoint_sha256'] == source['checkpoint_sha256']
            raw_ref = cached(raw_entry)
            teacher = PrefixTeacher()
            states = {name: PrefixRidge(12 if name == 'RGB_prefix' else 13) for name in LABELS}
            predictions = {name: np.empty(data['gt'].shape, dtype=np.float32) for name in LABELS}
            maximum = 0.; started = time.perf_counter()
            with torch.inference_mode():
                for t, low in enumerate(data['low']):
                    value = torch.from_numpy(low.transpose(2, 0, 1).copy()[None].astype(np.float32))/255
                    parts = reader.parts(value); core += 1
                    expected = torch.from_numpy(raw_ref[t].transpose(2, 0, 1).copy()[None])/255
                    maximum = max(maximum, float((parts['base'].clamp(0, 1)-expected).abs().max()))
                    assert maximum <= 2e-5
                    # Each prediction sees only coefficients fitted on strictly earlier frames.
                    for name in LABELS:
                        result = rgb_predict(parts['base'], states[name].physical) if name == 'RGB_prefix' else reader.predict(parts, states[name].physical)
                        assert torch.isfinite(result).all()
                        predictions[name][t] = result[0].clamp(0, 1).permute(1, 2, 0).contiguous().numpy()*255
                    if t < 8:
                        target = teacher.step(parts['base'])
                        bank = reader.basis(parts); rgb_bank = rgb_basis(parts['base'])
                        states['FST_prefix'].update(bank, parts['base'], target)
                        states['RGB_prefix'].update(rgb_bank, parts['base'], target)
                        # Explicit isolated oracle-like diagnostic: never a deployable target or shared state.
                        supervised = torch.from_numpy(data['gt'][t].transpose(2, 0, 1).copy()[None].astype(np.float32))/255
                        states['FST_GT_diagnostic'].update(bank, parts['base'], supervised)
                        fits += 3
                        append(out/'prefix-updates.jsonl', dict(seed=args.seed, scene=meta['scene'], frame=t,
                            updated_after_prediction=True, teacher_calls=teacher.count, teacher_flow_pairs=teacher.flow_pairs,
                            teacher_bytes=teacher.storage_bytes(), coefficients={name: state.physical.tolist() for name, state in states.items()},
                            state_bytes={name: state.storage_bytes() for name, state in states.items()}))
                        del target, bank, rgb_bank, supervised
                    else:
                        assert teacher.count == 8 and teacher.storage_bytes() == 0
                        assert all(state.count == 8 and state.gram is state.rhs is state.normalizer is None for state in states.values())
            append(out/'stage-cost.jsonl', dict(seed=args.seed, scene=meta['scene'], seconds=time.perf_counter()-started,
                shared_core_frames=len(data['low']), prefix_basis_frames=8, teacher_flow_pairs=7, ridge_updates=24,
                tail_calls=3*len(data['low'])+16, tail_calls_explanation='base/FST/GT diagnostic each frame, plus grouped bias and scale basis for8 prefix frames',
                state_bytes_per_deployable_route=dict(FST_prefix=52, RGB_prefix=48),
                scope='Shared experiment descriptive CPU work, not standalone method timing or handset latency.'))
            append(out/'source-parity.jsonl', dict(seed=args.seed, scene=meta['scene'], checkpoint_sha256=source['checkpoint_sha256'],
                maximum_normalized_difference_against_cached_raw=maximum))
            entries = []
            for name in LABELS:
                path = cache/f'{meta["scene"]}--{name}.npy'; np.save(path, predictions[name], allow_pickle=False)
                identity = dict(seed=args.seed, domain=meta['domain'], scene=meta['scene'], source_video=meta['source_video'],
                    label=name, parameters=4075, adaptive_state_values=12 if name == 'RGB_prefix' else 13,
                    project_updates=8000, deployable=name != 'FST_GT_diagnostic')
                entry = dict(**identity, path=str(path.relative_to(ROOT)), sha256=digest(path), shape=list(predictions[name].shape),
                    checkpoint_sha256=source['checkpoint_sha256'], frozen_coefficients=states[name].physical.tolist())
                entries.append(entry); append(out/'prediction_manifest.jsonl', entry)
            for ref in (raw_entry, rec_entry):
                entry = dict(ref, parameters=4075, project_updates=8000, deployable=True, adaptive_state_values=0)
                append(out/'prediction_manifest.jsonl', entry); entries.append(entry)
            for entry in entries:
                name = entry['label']; output = predictions[name] if name in predictions else (raw_ref if name == 'MobileIE_raw' else cached(rec_entry))
                for scope, start in (('full', 0), ('tail', 8)):
                    if scope == 'full' and name not in LABELS:
                        continue  # Original full reference metrics/JOD remain reused, never silently recomputed.
                    identity = {k: entry[k] for k in ('seed', 'domain', 'scene', 'source_video', 'label', 'parameters', 'project_updates', 'deployable')}
                    metrics, details = evaluate(output[start:], data['gt'][start:], data['reference_flow'][start:], data['reference_mask'][start:])
                    append(out/'per_sequence.jsonl', dict(**identity, scope=scope, **metrics))
                    for row in details:
                        append(out/'per_frame.jsonl', dict(**identity, scope=scope, **row))
                    rows += 1; frames += len(details)
            print(json.dumps(dict(seed=args.seed, scene=meta['scene'], new_quality_rows=rows, shared_core_frames=core)), flush=True)
            del data, predictions, raw_ref
        assert (rows, frames, core, fits) == (80, 3184, 448, 240)
        assert not torch.cuda.is_initialized(); verify(); complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(out/'runtime.json', dict(ended=stamp(), completed_normally=complete, seed=args.seed,
            new_quality_rows=rows, frame_rows=frames, shared_core_frames=core, ridge_updates=fits,
            global_SGD_updates=0, CUDA_initialized=torch.cuda.is_initialized(), seconds=time.perf_counter()-began))


if __name__ == '__main__':
    main()
