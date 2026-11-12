"""All20 registered static contacts from complete caches; no new enhancement."""
from datetime import datetime
import json
import time
import numpy as np
from PIL import Image, ImageDraw

from native_moment_core_r73 import BASE, ROOT
from check_native_moment_r73 import write
from dynamic_training_r34 import digest
from evaluate_native_moment_r73 import metas
from evaluate_cvvdp_references_r66 import cached
from prepare_cache_r5 import load_one
from summarize_native_moment_r73 import lines


def main():
    audit = json.loads((BASE/'report/exploration-r73-quality/audit.json').read_text()); assert audit['completed']
    entries = []
    for seed in [0, 1]: entries += lines(BASE/f'results/exploration-r73-eval-seed{seed}/prediction_manifest.jsonl')
    entries += lines(BASE/'process/exploration-r73/references001/prediction_manifest.jsonl')
    lookup = {(r['seed'], r['scene'], r['model'], r['history'], r['method']): r for r in entries}
    out = BASE/'report/exploration-r73-contacts'; out.mkdir(exist_ok=False)
    started = time.perf_counter(); sources = []
    for meta in metas()[0]:
        data = load_one(meta); h, w = data['low'].shape[1:3]; frames = [0, len(data['low'])//2, len(data['low'])-1]
        for seed in [0, 1]:
            selected = [lookup[(seed, meta['scene'])+condition] for condition in
                 [('R44_plain', 'current', 'recursive_self05'), ('spatial', 'current', 'raw'),
                  ('mean', 'causal', 'raw'), ('moment', 'causal', 'raw'), ('moment', 'current', 'raw')]]
            arrays = [data['low'], data['gt']] + [cached(r) for r in selected]
            labels = ['Input', 'GT', 'Old MSE rec .5', 'spatial raw', 'mean causal raw', 'moment causal raw', 'moment current raw']
            sheet = Image.new('RGB', (224*7, 940), 'white'); draw = ImageDraw.Draw(sheet)
            draw.text((6, 4), f'R73 {meta["scene"]} seed{seed} / final12000 / reused DEV / frames{frames}', fill='black')
            for col, (label, values) in enumerate(zip(labels, arrays)):
                draw.text((224*col+4, 24), label, fill='black')
                for row, frame in enumerate(frames):
                    image = Image.fromarray(np.rint(np.clip(values[frame], 0, 255)).astype(np.uint8))
                    y = 50+row*296; draw.text((224*col+4, y), f'Frame{frame} full FOV', fill='black')
                    sheet.paste(image.resize((224, 126), Image.Resampling.LANCZOS), (224*col, y+15))
                    draw.text((224*col+4, y+143), 'Native center128', fill='black')
                    sheet.paste(image.crop((w//2-64, h//2-64, w//2+64, h//2+64)), (224*col+48, y+158))
            path = out/f'fixed-{meta["scene"]}-seed{seed}.png'; sheet.save(path)
            sources.append(dict(scene=meta['scene'], seed=seed, frames=frames, path=str(path.relative_to(ROOT)),
                                sha256=digest(path), predictions=selected, actual_reviewed=False))
    assert len(sources) == 20
    write(out/'sources.json', sources)
    write(out/'runtime.json', dict(completed_normally=True, contacts=20, actual_reviewed=False,
          new_CNN=0, new_SGD=0, new_GPU=0, seconds=time.perf_counter()-started, ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__': main()
