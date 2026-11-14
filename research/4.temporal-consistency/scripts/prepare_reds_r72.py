"""Prepare the fixed eight native REDS low-light conditions, with no enhancement CNN."""
import hashlib
import json
from pathlib import Path
import time
import traceback

import cv2
import numpy as np
from PIL import Image
import torch

from exploration_r2_data import degrade, evaluate
from explore_state_r2 import forward_backward
from reds_references_r72 import BASE, ROOT, DATA, SCENES, append, check_sources, digest, frozen_sources, lines, stamp, write


def main():
    torch.set_num_threads(2); cv2.setNumThreads(1)
    assert not torch.cuda.is_initialized()
    source = BASE / 'process/exploration-r71/fetch001'
    terminal = json.loads((source / 'runtime.json').read_text())
    assert terminal['completed_normally'] and terminal['completed_members'] == terminal['validated_png_files_on_disk'] == 256
    review_path = BASE / 'report/exploration-r71-reds-sources/actual-visual-review.json'
    review = json.loads(review_path.read_text())
    assert review['actual_reviewed'] and review['contacts'] == 4
    records = lines(source / 'manifest.jsonl')
    assert len(records) == len({r['member'] for r in records}) == 256
    DATA.mkdir(parents=True, exist_ok=False)
    assets = ROOT / '.cache/experiments/exploration-r72/data001'
    assets.mkdir(parents=True, exist_ok=False)
    assert assets.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    frozen = frozen_sources()
    write(DATA / 'config.json', dict(started=stamp(), source_sha256=frozen,
          R71_manifest_sha256=digest(source / 'manifest.jsonl'), actual_review_sha256=digest(review_path),
          scenes=SCENES, frames_per_scene=64, photons=[20, 80], jpeg=False, source_fps=24,
          seed='SHA256 R72:{scene}:{photons}:0 first8 bytes little-endian, PCG64',
          CPU_Torch=2, OpenCV=1, CNN=0, SGD=0, GPU=0,
          scope='Custom synthetic low-light DEV from standard REDS sharp source; not natural night or official LLVE task.'))
    started = time.perf_counter(); complete = False; rows = []; decoded = 0; generated = 0

    def save(name, value):
        path = assets / (name + '.npy')
        np.save(path, value, allow_pickle=False)
        return dict(path=str(path.relative_to(ROOT)), sha256=digest(path), shape=list(value.shape), dtype=str(value.dtype))

    try:
        for scene in SCENES:
            members = sorted([r for r in records if Path(r['member']).parts[-2] == scene], key=lambda r: r['member'])
            assert len(members) == 64
            images = []
            for ordinal, record in enumerate(members):
                assert int(Path(record['member']).stem) == ordinal
                path = ROOT / record['path']; assert digest(path) == record['sha256']
                with Image.open(path) as image:
                    assert image.mode == 'RGB' and image.size == (1280, 720)
                    images.append(np.array(image, dtype=np.uint8))
                decoded += 1
            gt = np.stack(images); del images
            gt_source = save(scene + '--gt', gt)
            reference = [forward_backward(gt[t - 1], gt[t]) for t in range(1, 64)]
            flow = np.stack([r[0] for r in reference]); mask = np.stack([r[1] for r in reference]); del reference
            assert flow.shape == (63, 720, 1280, 2) and mask.shape == (63, 720, 1280)
            assert np.isfinite(flow).all() and mask.dtype == np.bool_
            motion = dict(reference_flow=save(scene + '--reference-flow', flow),
                          reference_mask=save(scene + '--reference-mask', mask))
            for photons in [20, 80]:
                seed = int.from_bytes(hashlib.sha256(f'R72:{scene}:{photons}:0'.encode()).digest()[:8], 'little')
                rng = np.random.Generator(np.random.PCG64(seed))
                low = np.stack([degrade(frame.astype(np.float32), rng, photons=photons, jpeg=False).astype(np.uint8)
                                for frame in gt])
                generated += len(low)
                assert low.dtype == np.uint8 and low.shape == (64, 720, 1280, 3)
                label = f'{scene}_photons{photons}'
                row = dict(domain='REDS_synthetic_DEV', scene=label, source_video=scene, photons=photons,
                           source_fps=24, input_rng_seed=seed, frames=64, gt=gt_source,
                           low=save(label + '--low', low), **motion)
                rows.append(row); write(DATA / (label + '.json'), row)
                scores, per_frame = evaluate(low.astype(np.float32), gt, flow, mask)
                identity = dict(domain=row['domain'], scene=label, source_video=scene,
                                photons=photons, family='Input', seed=None, method='input',
                                parameters=0, project_updates=0, future_wait_max=0)
                append(DATA / 'input_per_sequence.jsonl', dict(**identity, **scores))
                for frame in per_frame:
                    append(DATA / 'input_per_frame.jsonl', dict(**identity, **frame))
                print(json.dumps(dict(at=stamp(), scene=label, prepared_conditions=len(rows), CNN=0)), flush=True)
                del low
            del gt, flow, mask
        assert (len(rows), decoded, generated) == (8, 256, 512)
        assert not torch.cuda.is_initialized(); check_sources(frozen)
        write(DATA / 'index.json', dict(scenes=rows, source_fps=24, original_source_frames=256,
              degraded_observations=512, independent_source_videos=4,
              motion='GT-estimated full-FB only for metrics; deployment post uses own raw, not these fields',
              scope='Fixed development data, no model inference, source selection or independent final test.'))
        complete = True
    except BaseException:
        (DATA / 'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(DATA / 'runtime.json', dict(completed_normally=complete, conditions=len(rows),
              decoded_source_frames=decoded, degraded_observations=generated, input_rows=len(rows),
              CNN=0, SGD=0, GPU=0, seconds=time.perf_counter() - started, ended=stamp()))


if __name__ == '__main__':
    main()
