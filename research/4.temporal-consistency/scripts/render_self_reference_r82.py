"""Fixed first/middle/last images for every R82 seed and paired DEV condition."""
import json
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from self_reference_io_r82 import ROOT, BASE, digest, write, append, stamp
from self_reference_io_r82 import verify as verify_evaluation
from causal_reference_evaluation_r75 import metas, reference_predictions
from causal_reference_io_r75 import lines
from evaluate_cvvdp_references_r66 import cached
from prepare_cache_r5 import load_one
LABELS = ('spatial_recursive_self05', 'direct', 'centered', 'centered_wrong')


def main():
    verify_evaluation()
    audit = json.loads((BASE/'report/exploration-r82-joint/audit.json').read_text())
    assert audit['completed'] and audit['all_quality_rows'] == 140
    entries = []
    for seed in (0, 1):
        entries += lines(BASE/f'results/exploration-r82-eval-seed{seed}/prediction_manifest.jsonl')
    index = {(r['seed'], r['scene'], r['label']): r for r in entries}
    references = {(r['seed'], r['scene'], r['label']): r for r in reference_predictions()}
    scenes, _ = metas()
    out = BASE/'report/exploration-r82-contacts'; out.mkdir(exist_ok=False)
    started = time.perf_counter(); count = 0
    font = ImageFont.load_default(size=14)
    for meta in scenes:
        data = load_one(meta)
        times = [0, len(data['low'])//2, len(data['low'])-1]
        for seed in (0, 1):
            reference = references[seed, meta['scene'], 'MobileIE_recursive_self05']
            selected = [index[seed, meta['scene'], label] for label in LABELS]
            raw = references[seed, meta['scene'], 'MobileIE_raw']
            values = [('Input', data['low']), ('GT', data['gt']), ('MobileIE raw', cached(raw)), ('MobileIE rec .5', cached(reference))]
            values += [(entry['label'], cached(entry)) for entry in selected]
            canvas = Image.new('RGB', (2048, 554), 'white'); draw = ImageDraw.Draw(canvas)
            draw.text((8, 5), f"R82 seed {seed} / {meta['scene']} / fixed t={times}; static review only", fill='black', font=font)
            for column, (label, frames) in enumerate(values):
                for row, t in enumerate(times):
                    x, y = column*256, 36+row*172
                    draw.text((x+3, y), f'{label} t={t}', fill='black', font=font)
                    frame = Image.fromarray(np.clip(frames[t], 0, 255).astype(np.uint8))
                    frame.thumbnail((252, 144), Image.Resampling.LANCZOS)
                    canvas.paste(frame, (x+(256-frame.width)//2, y+23))
            path = out/f'seed{seed}--{meta["scene"]}.png'; canvas.save(path)
            append(out/'manifest.jsonl', dict(seed=seed, scene=meta['scene'], frames=times,
                path=str(path.relative_to(ROOT)), sha256=digest(path), sources=selected+[raw, reference], actual_reviewed=False))
            count += 1
        del data, values
    assert count == 20
    write(out/'runtime.json', dict(ended=stamp(), completed_normally=True, contacts=count,
        actual_reviewed=False, new_CNN=0, new_SGD=0, seconds=time.perf_counter()-started,
        scope='All fixed static images prepared; actual viewing must be recorded separately. Not playback or MOS.'))


if __name__ == '__main__':
    main()
