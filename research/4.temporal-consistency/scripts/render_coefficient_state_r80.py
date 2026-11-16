"""Fixed first/first-postwarm/last views of every seed and paired DEV condition."""
import json
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from coefficient_io_r80 import ROOT, BASE, LABELS, digest, write, append, lines, stamp, verify
from causal_reference_evaluation_r75 import metas
from evaluate_cvvdp_references_r66 import cached
from prepare_cache_r5 import load_one


def main():
    verify(); assert json.loads((BASE/'report/exploration-r80-joint/audit.json').read_text())['completed']
    entries = []
    for seed in (0, 1):
        entries += lines(BASE/f'results/exploration-r80-eval-seed{seed}/prediction_manifest.jsonl')
    index = {(r['seed'], r['scene'], r['label']):r for r in entries}; assert len(index) == 100
    scenes, _ = metas(); out = BASE/'report/exploration-r80-contacts'; out.mkdir(exist_ok=False)
    font = ImageFont.load_default(size=14); started = time.perf_counter(); count = 0
    labels = ('MobileIE_raw', 'MobileIE_recursive_self05', *LABELS)
    for meta in scenes:
        data = load_one(meta); times = [0, 8, len(data['low'])-1]
        for seed in (0, 1):
            selected = [index[seed, meta['scene'], label] for label in labels]
            views = [('Input', data['low']), ('GT', data['gt'])]
            views += [('GTfit(non-deploy)' if r['label'] == 'FST_GT_diagnostic' else r['label'], cached(r)) for r in selected]
            canvas = Image.new('RGB', (1792, 554), 'white'); draw = ImageDraw.Draw(canvas)
            draw.text((8, 5), f"R80 seed{seed} / {meta['scene']} / t={times}; GTfit is diagnostic; static images only", fill='black', font=font)
            for column, (label, values) in enumerate(views):
                for row, t in enumerate(times):
                    x, y = column*256, 36+row*172
                    draw.text((x+2, y), f'{label} t={t}', fill='black', font=font)
                    frame = Image.fromarray(np.clip(values[t], 0, 255).astype(np.uint8))
                    frame.thumbnail((252, 144), Image.Resampling.LANCZOS)
                    canvas.paste(frame, (x+(256-frame.width)//2, y+23))
            path = out/f'seed{seed}--{meta["scene"]}.png'; canvas.save(path)
            append(out/'manifest.jsonl', dict(seed=seed, scene=meta['scene'], frames=times,
                path=str(path.relative_to(ROOT)), sha256=digest(path), sources=selected, actual_reviewed=False))
            count += 1
        del data, views
    assert count == 20
    write(out/'runtime.json', dict(ended=stamp(), completed_normally=True, contacts=count, actual_reviewed=False,
        new_CNN=0, new_SGD=0, seconds=time.perf_counter()-started,
        scope='Generation only; actual review separately required. Not playback, native-detail inspection or MOS.'))


if __name__ == '__main__':
    main()
