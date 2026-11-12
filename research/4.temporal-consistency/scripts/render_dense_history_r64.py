"""All final DEV conditions/seeds, fixed frames; cached raw only, zero new CNN."""
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

from aggregation_controls_r20 import blend
from aligned_raw_history_r11 import align_history
from dynamic_training_r34 import BASE, ROOT, digest
from half_motion_r4 import half_forward_backward
from prepare_cache_r5 import load_one


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def recursive(raw):
    frames = [raw[0]]
    for t in range(1, len(raw)):
        flow, mask = half_forward_backward(raw[t-1], raw[t])
        moved, _ = align_history(frames[-1], raw[t], flow, mask)
        frames.append(blend(raw[t], moved, .5))
    value = np.stack(frames)
    assert np.array_equal(value[0], raw[0]) and np.isfinite(value).all()
    return value


def main():
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    assert not torch.cuda.is_initialized()
    folder = BASE/'report/exploration-r64-dense-history'
    audit = json.loads((folder/'audit.json').read_text())
    assert audit['processing_complete'] and audit['new_quality_rows'] == 140
    coordinator = BASE/'process/exploration-r64/finalize001/runtime.json'
    assert json.loads(coordinator.read_text())['completed_normally']
    manifests = {}
    for seed in [0, 1]:
        run = BASE/f'results/exploration-r64-stage001-eval-seed{seed}'
        runtime = json.loads((run/'runtime.json').read_text())
        assert runtime['completed_normally'] and runtime['rows'] == 70
        assert runtime['neural_sequences'] == 40 and runtime['center_outputs'] == 1792
        path = run/'prediction_manifest.jsonl'
        entries = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(entries) == 40
        manifests[seed] = (path, {(e['scene'], e['model'], e['history']): e for e in entries})
    metas = json.loads((BASE/'results/exploration-r30-cache/index.json').read_text())['scenes']
    metas += [{**r, 'domain': 'paired_real_ND_DEV', 'source_video': r['scene']}
              for r in json.loads((BASE/'results/exploration-r10-cache/index.json').read_text())['scenes']
              if r['domain'] == 'native_crop']
    assert len(metas) == 10
    process = BASE/'process/exploration-r64/render001'
    process.mkdir(exist_ok=False)
    write(process/'config.json', {'at': datetime.now().astimezone().isoformat(),
        'protocol_sha256': digest(BASE/'docs/exploration-r64-final-visual-protocol.md'),
        'source_sha256': {n: digest(BASE/'scripts'/n) for n in
            ['render_dense_history_r64.py', 'half_motion_r4.py', 'aligned_raw_history_r11.py',
             'aggregation_controls_r20.py', 'prepare_cache_r5.py']},
        'summary_audit_sha256': digest(folder/'audit.json'),
        'coordinator_runtime_sha256': digest(coordinator), 'new_CNN_sequences': 0,
        'scope': 'Cached final raw and exact classic-post re-render only, no new quality or selection.'})
    font = ImageFont.truetype(str(Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'), 14)
    started, complete, records = time.perf_counter(), False, []
    try:
        for seed in [0, 1]:
            manifest, lookup = manifests[seed]
            for meta in metas:
                data = load_one(meta)
                expected = 64 if meta['domain'] == 'davis_train_pilot' else 16
                assert len(data['low']) == expected and data['low'].shape == data['gt'].shape
                arrays, labels = [data['low'], data['gt']], ['Input', 'GT']
                sources = []
                for name, route in [('plain', 'current'), ('spatial', 'current'),
                                    ('causal', 'causal'), ('causal', 'current')]:
                    entry = lookup[meta['scene'], name, route]
                    path = ROOT/entry['path']
                    assert path.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
                    assert digest(path) == entry['sha256']
                    raw = np.load(path, mmap_mode='r', allow_pickle=False)
                    assert raw.shape == data['low'].shape and list(raw.shape) == entry['shape']
                    assert np.isfinite(raw).all()
                    sources.append(entry)
                    arrays.append(raw)
                    labels.append('causal current' if name == 'causal' and route == 'current' else name+' raw')
                    if not (name == 'causal' and route == 'current'):
                        arrays.append(recursive(raw))
                        labels.append(name+' recursive')
                assert len(arrays) == 9
                frames = [0, expected//2, expected-1]
                height, width = data['low'].shape[1:3]
                image = Image.new('RGB', (9*width, 58+3*(height+24)), 'white')
                draw = ImageDraw.Draw(image)
                draw.text((8, 3), f"R64 {meta['scene']} / seed{seed} / fixed frames {frames} / final12000 / reused paired DEV", font=font, fill='black')
                for col, label in enumerate(labels):
                    draw.text((col*width+6, 29), label, font=font, fill='black')
                for row, frame in enumerate(frames):
                    y = 58+row*(height+24)
                    draw.text((6, y), f'Frame {frame}', font=font, fill='black')
                    for col, array in enumerate(arrays):
                        display = np.rint(np.clip(array[frame], 0, 255)).astype(np.uint8)
                        image.paste(Image.fromarray(display), (col*width, y+24))
                path = folder/f"fixed-{meta['scene']}-seed{seed}.png"
                assert not path.exists()
                image.save(path)
                records.append({'domain': meta['domain'], 'scene': meta['scene'], 'seed': seed,
                    'frames': frames, 'labels': labels, 'contact': str(path.relative_to(ROOT)),
                    'sha256': digest(path), 'manifest_sha256': digest(manifest), 'predictions': sources,
                    'actual_reviewed': False})
                print(json.dumps({'seed': seed, 'scene': meta['scene'], 'contacts': len(records)}), flush=True)
        assert len(records) == 20 and not torch.cuda.is_initialized()
        write(folder/'visual_sources.json', {'created_at': datetime.now().astimezone().isoformat(),
            'contacts': records, 'new_CNN_sequences': 0, 'new_SGD_updates': 0, 'new_source_pixels': 0,
            'scope': 'All conditions/seeds; display rounding only; static three-frame contacts are not full playback or MOS.'})
        complete = True
    except BaseException:
        (process/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(process/'runtime.json', {'completed_normally': complete, 'contacts': len(records),
            'new_CNN_sequences': 0, 'new_SGD_updates': 0, 'cuda_initialized': torch.cuda.is_initialized(),
            'seconds': time.perf_counter()-started, 'ended_at': datetime.now().astimezone().isoformat()})


if __name__ == '__main__':
    main()
