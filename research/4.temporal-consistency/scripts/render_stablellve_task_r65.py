"""All20 fixed contacts after complete quality; full FOV and native center128."""
import json
import time
import traceback
from pathlib import Path
from datetime import datetime

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from check_stablellve_task_r65 import write
from dynamic_training_r34 import BASE, ROOT, digest
from prepare_cache_r5 import load_one
from render_dense_history_r64 import recursive


def lines(path):
    return [json.loads(row) for row in path.read_text().splitlines()]


def cached(entry):
    path = ROOT/entry['path']
    assert path.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    assert digest(path) == entry['sha256']
    values = np.load(path, mmap_mode='r', allow_pickle=False)
    assert list(values.shape) == entry['shape']
    return values


def main():
    torch.set_num_threads(2); cv2.setNumThreads(1); assert not torch.cuda.is_initialized()
    folder = BASE/'report/exploration-r65-stablellve-task'
    audit = json.loads((folder/'audit.json').read_text())
    assert audit['complete_paired_table'] and audit['new_quality_rows'] == 80 and audit['all_condition_rows'] == 220
    coordinator = BASE/'process/exploration-r65/finalize001/runtime.json'
    assert json.loads(coordinator.read_text())['completed_normally']
    indices = [BASE/f'results/{name}/index.json' for name in ['exploration-r30-cache', 'exploration-r10-cache']]
    metas = json.loads(indices[0].read_text())['scenes']
    metas += [{**r, 'domain': 'paired_real_ND_DEV'} for r in json.loads(indices[1].read_text())['scenes']
              if r['domain'] == 'native_crop']
    assert len(metas) == 10
    original = BASE/'results/exploration-r62-stablellve-common-attempt001/prediction_manifest.jsonl'
    published = {r['scene']: r for r in lines(original) if r['method'] == 'raw'}
    process = BASE/'process/exploration-r65/render001'; process.mkdir(exist_ok=False)
    write(process/'config.json', dict(at=datetime.now().astimezone().isoformat(),
          protocol_sha256=digest(BASE/'docs/exploration-r65-final-visual-protocol.md'),
          source_sha256={n: digest(BASE/'scripts'/n) for n in ['render_stablellve_task_r65.py',
              'render_dense_history_r64.py', 'half_motion_r4.py', 'aligned_raw_history_r11.py', 'aggregation_controls_r20.py']},
          coordinator_runtime_sha256=digest(coordinator), new_CNN=0, new_SGD=0, new_GPU=0,
          all_sources_and_seeds=True, original_published_manifest_sha256=digest(original), actual_reviewed=False))
    font = ImageFont.truetype(str(Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'), 14)
    labels = ['Input', 'GT', 'R65 L1 raw', 'R65 L1 rec .5', 'R65 commute20 raw',
              'R65 commute20 rec .5', 'R44 MIE raw', 'R44 MIE rec .5', 'R62 clean raw (0upd)']
    start, complete, contacts = time.perf_counter(), False, []
    try:
        for seed in [0, 1]:
            manifest = BASE/f'results/exploration-r65-stage001-eval-seed{seed}/prediction_manifest.jsonl'
            current = {(r['scene'], r['arm'], r['method']): r for r in lines(manifest)}
            reference = BASE/f'results/exploration-r44-stage001-eval-seed{seed}/prediction_manifest.jsonl'
            mie = {r['scene']: r for r in lines(reference) if r['model'] == 'square_mse'}
            assert len(current) == 40 and len(mie) == len(published) == 10
            for meta in metas:
                data = load_one(meta); shape = data['low'].shape
                assert shape == data['gt'].shape and shape[0] in (16, 64)
                frames = [0, shape[0]//2, shape[0]-1]
                arrays, sources = [data['low'], data['gt']], []
                for arm in ['spatial_l1', 'commute20']:
                    for method in ['raw', 'recursive_self05']:
                        entry = current[meta['scene'], arm, method]; arrays.append(cached(entry)); sources.append(entry)
                entry = mie[meta['scene']]; raw = cached(entry)
                arrays += [raw, recursive(raw)]; sources.append(dict(**entry, displayed_methods=['raw', 'recomputed recursive .5']))
                entry = published[meta['scene']]; arrays.append(cached(entry)); sources.append(entry)
                assert len(arrays) == 9 and all(a.shape == shape for a in arrays)
                y, x = (shape[1]-128)//2, (shape[2]-128)//2
                image = Image.new('RGB', (9*266, 58+3*318), 'white'); draw = ImageDraw.Draw(image)
                draw.text((8, 3), f'R65 {meta["scene"]} / seed{seed} / final8000 / frames{frames} / reused DEV / native center128', font=font, fill='black')
                for col, label in enumerate(labels): draw.text((col*266+4, 29), label, font=font, fill='black')
                for row, frame in enumerate(frames):
                    top = 58+row*318
                    draw.text((4, top), f'Frame{frame} full FOV', font=font, fill='black')
                    draw.text((4, top+174), f'Frame{frame} native center128', font=font, fill='black')
                    for col, array in enumerate(arrays):
                        value = array[frame]; assert np.isfinite(value).all()
                        display = Image.fromarray(np.rint(np.clip(value, 0, 255)).astype(np.uint8))
                        fitted = display.copy(); fitted.thumbnail((266, 150), Image.Resampling.LANCZOS)
                        image.paste(fitted, (col*266+(266-fitted.width)//2, top+24+(150-fitted.height)//2))
                        image.paste(display.crop((x, y, x+128, y+128)), (col*266+69, top+190))
                path = folder/f'fixed-{meta["scene"]}-seed{seed}.png'; assert not path.exists(); image.save(path)
                contacts.append(dict(scene=meta['scene'], domain=meta['domain'], seed=seed, frames=frames,
                    labels=labels, center_xy=[x, y], center_shape=[128, 128], full_FOV_display_box=[266, 150],
                    input_shape=list(shape), input_cache=meta, predictions=sources, actual_reviewed=False,
                    contact=str(path.relative_to(ROOT)), sha256=digest(path), bytes=path.stat().st_size,
                    current_manifest_sha256=digest(manifest), R44_manifest_sha256=digest(reference)))
                print(json.dumps(dict(scene=meta['scene'], seed=seed, contacts=len(contacts))), flush=True)
                del data, arrays, raw
        assert len(contacts) == 20 and not torch.cuda.is_initialized()
        write(folder/'visual_sources.json', dict(contacts=contacts, actual_reviewed=False,
              scope='Generated fixed stills only; full-FOV resize weakens noise, native center does not cover all scene; no video/MOS.'))
        complete = True
    except BaseException:
        (process/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(process/'runtime.json', dict(completed_normally=complete, contacts=len(contacts),
              new_CNN=0, new_SGD=0, new_GPU=0, actual_reviewed=False,
              seconds=time.perf_counter()-start, ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__': main()
