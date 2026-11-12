"""Inspect all paired ND sources and seeds using existing final raw predictions."""
import json
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dynamic_training_r34 import BASE, ROOT, digest
from prepare_cache_r5 import load_one


def main():
    folder = BASE/'report/exploration-r44-bounded-video'
    assert json.loads((folder/'audit.json').read_text())['new_rows'] == 240
    metas = [m for m in json.loads((BASE/'results/exploration-r10-cache/index.json').read_text())['scenes']
             if m['domain'] == 'native_crop']
    assert len(metas) == 4
    font = ImageFont.truetype(str(Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'), 15)
    models = ['square_mse', 'square_temporal', 'bounded_mse', 'bounded_temporal']
    records = []
    for seed in [0, 1]:
        manifest = BASE/f'results/exploration-r44-stage001-eval-seed{seed}/prediction_manifest.jsonl'
        values = [json.loads(l) for l in manifest.read_text().splitlines()]
        lookup = {(v['scene'], v['model']): v for v in values}
        for meta in metas:
            data = load_one(meta)
            predictions, sources = [], []
            for model in models:
                entry = lookup[(meta['scene'], model)]
                path = ROOT/entry['path']
                assert path.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
                raw = np.load(path, mmap_mode='r', allow_pickle=False)
                assert list(raw.shape) == entry['shape'] and len(raw) == 16
                predictions.append(raw)
                sources.append({'model': model, 'path': entry['path'],
                                'recorded_prediction_sha256': entry['sha256']})
            arrays = [data['low'], data['gt'], *predictions]
            h, w = arrays[0].shape[1:3]
            canvas = Image.new('RGB', (6*w, 46+3*(h+22)), 'white')
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 3), f"R44 {meta['scene']} / seed {seed}; fixed frames 0, 8, 15; raw only; paired ND DEV", font=font, fill='black')
            for column, label in enumerate(['Input', 'Reference', *models]):
                draw.text((column*w+8, 25), label, font=font, fill='black')
            for row, frame in enumerate([0, 8, 15]):
                y = 46+row*(h+22)
                draw.text((8, y), f'Frame {frame}', font=font, fill='black')
                for column, array in enumerate(arrays):
                    value = np.rint(np.clip(array[frame], 0, 255)).astype(np.uint8)
                    canvas.paste(Image.fromarray(value), (column*w, y+22))
            target = folder/f"real-{meta['scene']}-seed{seed}.png"
            canvas.save(target)
            records.append({'scene': meta['scene'], 'seed': seed, 'frames': [0, 8, 15],
                            'contact_sheet': str(target.relative_to(ROOT)),
                            'manifest_sha256': digest(manifest), 'predictions': sources})
    (folder/'visual_sources.json').write_text(json.dumps({
        'new_neural_sequences': 0, 'contacts': records,
        'scope': 'All four existing paired daytime ND DEV clips, both seeds, all raw arms. '
                 'Display rounding only; metrics remain floating point. No nighttime, perceived-flicker or phone claim.'
    }, indent=2)+'\n')


if __name__ == '__main__':
    main()
