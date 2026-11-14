"""Twenty preassigned classical-reference contacts; no new enhancement inference."""
from datetime import datetime
import json
from pathlib import Path
import time
import traceback

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from check_spatial_l1_r69 import write
from dynamic_training_r34 import BASE, ROOT, digest
from prepare_cache_r5 import load_one
from render_stablellve_task_r65 import cached, lines
from render_dense_history_r64 import recursive


def main():
    torch.set_num_threads(2); cv2.setNumThreads(1); assert not torch.cuda.is_initialized()
    folder = BASE/'report/exploration-r70-classical-quality'
    audit = json.loads((folder/'audit.json').read_text())
    assert audit['completed'] and audit['new_quality_rows'] == 80
    indices = [BASE/f'results/{name}/index.json' for name in ['exploration-r30-cache', 'exploration-r10-cache']]
    metas = json.loads(indices[0].read_text())['scenes']
    metas += [{**r, 'domain': 'paired_real_ND_DEV'} for r in json.loads(indices[1].read_text())['scenes'] if r['domain'] == 'native_crop']
    assert len(metas) == 10
    process = BASE/'process/exploration-r70/render001'; process.mkdir(exist_ok=False)
    names = ['render_classical_block_r70.py', 'render_stablellve_task_r65.py', 'render_dense_history_r64.py',
             'half_motion_r4.py', 'aligned_raw_history_r11.py', 'aggregation_controls_r20.py']
    write(process/'config.json', dict(at=datetime.now().astimezone().isoformat(),
          protocol_sha256=digest(BASE/'docs/exploration-r70-classical-block-reference-protocol.md'),
          quality_audit_sha256=digest(folder/'audit.json'),
          source_sha256={name: digest(BASE/'scripts'/name) for name in names},
          new_CNN=0, new_SGD=0, new_GPU=0, actual_reviewed=False, all_sources_and_seeds=True))
    font = ImageFont.truetype(str(Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'), 14)
    labels = ['Input', 'GT', 'R44 MIE MSE raw', 'R44 MIE MSE rec .5',
              'R70 spatial-only raw', 'R70 spatial-only rec .5', 'R70 video-block raw', 'R70 video-block rec .5']
    start, complete, contacts = time.perf_counter(), False, []
    try:
        for seed in [0, 1]:
            paths = [BASE/'results/exploration-r70-classical-attempt001/prediction_manifest.jsonl',
                     BASE/f'results/exploration-r44-stage001-eval-seed{seed}/prediction_manifest.jsonl']
            records = [r for r in lines(paths[0]) if r['seed'] == seed]
            spatial = {(r['scene'], r['method']): r for r in records if r['mode'] == 'spatial_only'}
            video = {(r['scene'], r['method']): r for r in records if r['mode'] == 'video_default'}
            mse = {r['scene']: r for r in lines(paths[1]) if r['model'] == 'square_mse'}
            assert (len(spatial), len(mse), len(video)) == (20, 10, 20)
            for meta in metas:
                data = load_one(meta); shape = data['low'].shape; frames = [0, shape[0]//2, shape[0]-1]
                assert shape == data['gt'].shape and shape[0] in (16, 64)
                entry = mse[meta['scene']]; raw = cached(entry)
                arrays = [data['low'], data['gt'], raw, recursive(raw)]
                sources = [dict(**entry, displayed_methods=['raw', 'recomputed own-raw recursive .5'])]
                for records in [spatial, video]:
                    for method in ['raw', 'recursive_self05']:
                        entry = records[meta['scene'], method]; arrays.append(cached(entry)); sources.append(entry)
                assert len(arrays) == 8 and all(array.shape == shape for array in arrays)
                y, x = (shape[1]-128)//2, (shape[2]-128)//2
                image = Image.new('RGB', (8*266, 58+3*318), 'white'); draw = ImageDraw.Draw(image)
                draw.text((8, 3), f'R70 {meta["scene"]} / seed{seed} / final8000 / frames{frames} / reused DEV / native center128', font=font, fill='black')
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
                      manifests={str(path.relative_to(ROOT)): digest(path) for path in paths}))
                print(json.dumps(dict(scene=meta['scene'], seed=seed, contacts=len(contacts))), flush=True)
                del data, arrays, raw
        assert len(contacts) == 20 and not torch.cuda.is_initialized()
        write(folder/'visual_sources.json', dict(contacts=contacts, actual_reviewed=False,
              scope='Generated static contacts only; resized full FOV plus native center, no full-video viewing/MOS.'))
        complete = True
    except BaseException:
        (process/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(process/'runtime.json', dict(completed_normally=complete, contacts=len(contacts), new_CNN=0,
              new_SGD=0, new_GPU=0, actual_reviewed=False, seconds=time.perf_counter()-start,
              ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__': main()
