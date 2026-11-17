"""Four registered static contacts; full metrics cover all conditions."""
import json
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from shared_geometry_io_r88 import ROOT, BASE, PROCESS, digest, write, append, stamp, verify
from causal_reference_evaluation_r75 import metas, reference_predictions
from causal_reference_io_r75 import lines
from evaluate_cvvdp_references_r66 import cached
from prepare_cache_r5 import load_one
LABELS = ('plain_true', 'complement_true')


def main():
    verify(json.loads((PROCESS/'evaluation-source-freeze001.json').read_text()))
    audit = json.loads((BASE/'report/exploration-r88-joint/audit.json').read_text())
    assert audit['completed'] and audit['all_quality_rows'] == 140
    entries = []
    for seed in (0, 1):
        entries += lines(BASE/f'results/exploration-r88-eval-seed{seed}/prediction_manifest.jsonl')
    index = {(r['seed'], r['scene'], r['label']): r for r in entries}
    references = {(r['seed'], r['scene'], r['label']): r for r in reference_predictions()}
    dis = {(r['seed'], r['scene']): r for r in lines(BASE/'results/exploration-r86-eval001/prediction_manifest.jsonl')}
    scenes, _ = metas()
    scenes = [r for r in scenes if r['scene'] in ('bear_photons20', 'video_47')]
    assert len(scenes) == 2
    out = BASE/'report/exploration-r88-contacts'; out.mkdir(exist_ok=False)
    started = time.perf_counter(); count = 0
    font = ImageFont.load_default(size=14)
    for meta in scenes:
        data = load_one(meta)
        times = [0, len(data['low'])//2, len(data['low'])-1]
        for seed in (0, 1):
            reference = references[seed, meta['scene'], 'MobileIE_recursive_self05']
            selected = [index[seed, meta['scene'], label] for label in LABELS]
            raw = references[seed, meta['scene'], 'MobileIE_raw']
            values = [('Input', data['low']), ('GT', data['gt']), ('MobileIE raw', cached(raw)), ('MobileIE rec .5', cached(reference)), ('DIS rec .5', cached(dis[seed, meta['scene']]))]
            values += [(entry['label'], cached(entry)) for entry in selected]
            canvas = Image.new('RGB', (1792, 554), 'white'); draw = ImageDraw.Draw(canvas)
            draw.text((8, 5), f"R88 seed {seed} / {meta['scene']} / fixed t={times}; static review only", fill='black', font=font)
            for column, (label, frames) in enumerate(values):
                for row, t in enumerate(times):
                    x, y = column*256, 36+row*172
                    draw.text((x+3, y), f'{label} t={t}', fill='black', font=font)
                    frame = Image.fromarray(np.clip(frames[t], 0, 255).astype(np.uint8))
                    frame.thumbnail((252, 144), Image.Resampling.LANCZOS)
                    canvas.paste(frame, (x+(256-frame.width)//2, y+23))
            path = out/f'seed{seed}--{meta["scene"]}.png'; canvas.save(path)
            append(out/'manifest.jsonl', dict(seed=seed, scene=meta['scene'], frames=times,
                path=str(path.relative_to(ROOT)), sha256=digest(path), sources=selected+[raw, reference, dis[seed, meta['scene']]], actual_reviewed=False))
            count += 1
        del data, values
    assert count == 4
    write(out/'runtime.json', dict(ended=stamp(), completed_normally=True, contacts=count,
        actual_reviewed=False, new_CNN=0, new_SGD=0, seconds=time.perf_counter()-started,
        scope='Four preregistered source/seed static images prepared; actual viewing must be recorded separately. Not playback or MOS.'))


if __name__ == '__main__':
    main()
