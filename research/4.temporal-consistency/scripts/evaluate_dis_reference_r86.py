"""One artificial qualification then all twenty cached-F0 classical DIS outputs."""
import os
if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise SystemExit('CPU entry requires an initially empty CUDA mask')

import json
import time
import traceback

import cv2
import numpy as np
import torch

from dis_reference_io_r86 import ROOT, BASE, PROCESS, QUALITY, LABEL, verify, cached, write, append, stamp, digest
from fixed_dis_r86 import FixedDIS, RecursivePipeline, qualification
from causal_reference_evaluation_r75 import metas, reference_predictions
from prepare_cache_r5 import load_one
from exploration_r2_data import evaluate


def main():
    frozen = verify(); torch.set_num_threads(2); cv2.setNumThreads(1)
    QUALITY.mkdir(exist_ok=False)
    assets = ROOT/'.cache/experiments/exploration-r86/eval001'
    assert assets.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    assets.mkdir(parents=True, exist_ok=False)
    scenes, indices = metas()
    references = {(v['seed'], v['scene'], v['label']): v for v in reference_predictions()}
    write(QUALITY/'config.json', dict(at=stamp(), pid=os.getpid(), frozen=frozen, indices=indices,
        expected_quality_rows=20, expected_frame_rows=896, expected_DIS_quality_pairs=876,
        expected_DIS_toy_pairs=2, new_enhancer_CNN=0, new_SGD=0,
        scope='All repeated DEV, fixed known classical native DIS preset. Not a new learned method or phone result.'))
    started = time.perf_counter(); complete = False
    pairs = rows = frame_rows = toy_pairs = 0
    try:
        gate = qualification(); toy_pairs = gate['DIS_bidirectional_pairs']
        write(QUALITY/'qualification.json', dict(at=stamp(), **gate))
        for seed in (0, 1):
            for meta in scenes:
                ref = references[seed, meta['scene'], 'MobileIE_raw']
                raw = cached(ref); data = load_one(meta)
                assert raw.shape == data['gt'].shape
                motion = FixedDIS(); stream = RecursivePipeline(motion)
                output = np.empty(raw.shape, np.float32)
                for t, current in enumerate(raw):
                    value, field = stream.step(np.asarray(current, dtype=np.float32)); output[t] = value
                    if field is not None:
                        flow, mask = field; pairs += 1
                        append(QUALITY/'deployment-pairs.jsonl', dict(seed=seed, scene=meta['scene'],
                            frame=t, accepted_fraction=float(mask.mean()),
                            mean_native_flow_norm=float(np.linalg.norm(flow, axis=2).mean())))
                assert motion.pairs == len(raw)-1
                assert np.array_equal(output[0], raw[0]) and np.isfinite(output).all()
                metrics, details = evaluate(output, data['gt'], data['reference_flow'], data['reference_mask'])
                identity = dict(seed=seed, domain=meta['domain'], scene=meta['scene'],
                    source_video=meta['source_video'], label=LABEL, parameters=4075,
                    new_parameters=0, method='OpenCV4.11 native DIS ULTRAFAST / legacy FB / recursive05')
                path = assets/f'seed{seed}--{meta["scene"]}.npy'; np.save(path, output, allow_pickle=False)
                append(QUALITY/'prediction_manifest.jsonl', dict(**identity, path=str(path.relative_to(ROOT)),
                    shape=list(output.shape), sha256=digest(path), dtype='float32',
                    checkpoint_sha256=ref['checkpoint_sha256'], frozen_input=ref,
                    OpenCV_parameters=motion.parameters))
                append(QUALITY/'per_sequence.jsonl', dict(**identity, **metrics)); rows += 1
                for row in details:
                    append(QUALITY/'per_frame.jsonl', dict(**identity, **row)); frame_rows += 1
                print(json.dumps(dict(seed=seed, scene=meta['scene'], quality_rows=rows,
                    frame_rows=frame_rows, DIS_quality_pairs=pairs)), flush=True)
                del data, raw, output, stream, motion
        assert (rows, frame_rows, pairs, toy_pairs) == (20, 896, 876, 2)
        assert not torch.cuda.is_initialized(); verify(); complete = True
    except BaseException:
        (QUALITY/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(QUALITY/'runtime.json', dict(ended=stamp(), pid=os.getpid(), completed_normally=complete,
            seconds=time.perf_counter()-started, quality_rows=rows, frame_rows=frame_rows,
            DIS_quality_pairs=pairs, DIS_toy_pairs=toy_pairs, new_enhancer_CNN=0, new_SGD=0,
            CUDA_initialized=torch.cuda.is_initialized()))


if __name__ == '__main__':
    main()
