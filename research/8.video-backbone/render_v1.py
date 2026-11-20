"""Four preselected static DEV contacts, no extra enhancement forward passes."""
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from runtime_io import ROOT, BASE, write, stamp, digest, verify
from causal_reference_evaluation_r75 import metas, reference_predictions
from prepare_cache_r5 import load_one


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def array(entry):
    path = ROOT / entry['path']
    assert path.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    assert digest(path) == entry['sha256']
    return np.load(path, mmap_mode='r', allow_pickle=False)


def main():
    verify()
    assert json.loads((BASE / 'report/v1-joint/audit.json').read_text())['completed']
    entries = []
    for seed in (0, 1):
        entries += lines(BASE / f'results/v1-quality-seed{seed}/prediction_manifest.jsonl')
    old = reference_predictions()
    dis = lines(ROOT / 'research/4.temporal-consistency/results/exploration-r86-eval001/prediction_manifest.jsonl')
    scenes, _ = metas()
    out = BASE / 'report/v1-contacts'
    out.mkdir(parents=True, exist_ok=False)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
    columns = [('Input', None), ('GT', None), ('F0', 'MobileIE_raw'),
               ('F0 + FB', 'MobileIE_recursive_self05'), ('F0 + DIS', 'DIS_recursive_self05'),
               ('Packed spatial', 'topic8_spatial_raw'), ('Spatial + DIS', 'topic8_spatial_raw_DIS05'),
               ('Fixed pre-fusion', 'topic8_fixed_causal'), ('Learned pre-fusion', 'topic8_learned_causal')]
    records = []
    for scene in ('bear_photons20', 'video_47'):
        meta = next(r for r in scenes if r['scene'] == scene)
        data = load_one(meta)
        times = [0, len(data['gt']) // 2, len(data['gt']) - 1]
        for seed in (0, 1):
            chosen = {(r['label']): r for r in entries + old + dis if r['seed'] == seed and r['scene'] == scene}
            canvas = Image.new('RGB', (9 * 256, 32 + len(times) * 168), 'white')
            draw = ImageDraw.Draw(canvas)
            for col, (title, label) in enumerate(columns):
                draw.text((col * 256 + 4, 8), title, font=font, fill='black')
                values = data['low'] if title == 'Input' else data['gt'] if title == 'GT' else array(chosen[label])
                for row, index in enumerate(times):
                    tile = Image.fromarray(np.clip(np.rint(values[index]), 0, 255).astype(np.uint8))
                    tile.thumbnail((256, 144), Image.Resampling.LANCZOS)
                    canvas.paste(tile, (col * 256, 32 + row * 168))
                    draw.text((col * 256 + 4, 177 + row * 168), f't={index}', font=font, fill='black')
            path = out / f'seed{seed}--{scene}.png'
            canvas.save(path)
            records.append(dict(seed=seed, scene=scene, indices=times,
                                path=str(path.relative_to(ROOT)), sha256=digest(path)))
    write(out / 'render.json', dict(at=stamp(), completed=True, images=records,
          actual_viewed=False, new_backbone_images=0,
          scope='Only two preselected sources, two seeds, three stills; not playback, MOS, or independent final.'))


if __name__ == '__main__':
    main()
