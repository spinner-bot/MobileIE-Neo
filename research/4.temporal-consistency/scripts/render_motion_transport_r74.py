"""All20 fixed R74 contacts from completed caches; actual review is separate."""
from datetime import datetime
import json
from pathlib import Path
import time

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from motion_transport_r74 import BASE, ROOT
from check_motion_transport_r74 import write
from dynamic_training_r34 import digest
from evaluate_motion_transport_r74 import metas, reference_predictions
from evaluate_cvvdp_references_r66 import cached, lines
from prepare_cache_r5 import load_one


def main():
    audit = json.loads((BASE/'report/exploration-r74-quality/audit.json').read_text()); assert audit['completed']
    entries, _ = reference_predictions()
    for seed in [0, 1]: entries += lines(BASE/f'results/exploration-r74-eval-seed{seed}/prediction_manifest.jsonl')
    lookup = {(r['seed'], r['scene'], r['model'], r['history'], r['method']): r for r in entries}
    out = BASE/'report/exploration-r74-contacts'; out.mkdir(exist_ok=False)
    font = ImageFont.truetype(str(Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'), 14)
    started, complete, sources = time.perf_counter(), False, []
    try:
        for meta in metas()[0]:
            data = load_one(meta); h, w = data['low'].shape[1:3]; frames = [0, len(data['low'])//2, len(data['low'])-1]
            for seed in [0, 1]:
                selected = [lookup[(seed, meta['scene'])+condition] for condition in
                    [('R44_plain', 'current', 'raw'), ('R44_plain', 'current', 'recursive_self05'),
                     ('task', 'causal', 'raw'), ('proxy', 'causal', 'raw'), ('proxy', 'zero_flow', 'raw')]]
                arrays = [data['low'], data['gt']]+[cached(r) for r in selected]
                labels = ['Input', 'GT', 'Frozen raw', 'Old rec .5', 'task causal', 'proxy causal', 'proxy zero-flow']
                sheet = Image.new('RGB', (224*7, 956), 'white'); draw = ImageDraw.Draw(sheet)
                draw.text((6, 4), f'R74 {meta["scene"]} seed{seed} / spatial8000 motion4000 / reused DEV / frames{frames}', font=font, fill='black')
                for col, (label, values) in enumerate(zip(labels, arrays)):
                    draw.text((224*col+4, 28), label, font=font, fill='black')
                    for row, frame in enumerate(frames):
                        picture = Image.fromarray(np.rint(np.clip(values[frame], 0, 255)).astype(np.uint8))
                        y = 56+row*296
                        draw.text((224*col+4, y), f'Frame{frame} full FOV', font=font, fill='black')
                        sheet.paste(picture.resize((224, 126), Image.Resampling.LANCZOS), (224*col, y+20))
                        draw.text((224*col+4, y+148), 'Native center128', font=font, fill='black')
                        sheet.paste(picture.crop((w//2-64, h//2-64, w//2+64, h//2+64)), (224*col+48, y+166))
                path = out/f'fixed-{meta["scene"]}-seed{seed}.png'; sheet.save(path)
                sources.append(dict(scene=meta['scene'], seed=seed, frames=frames, path=str(path.relative_to(ROOT)),
                      sha256=digest(path), predictions=selected, actual_reviewed=False))
        assert len(sources) == 20; write(out/'sources.json', sources); complete = True
    finally:
        write(out/'runtime.json', dict(completed_normally=complete, contacts=len(sources), actual_reviewed=False,
              new_enhancement_CNN=0, new_motion_CNN=0, new_SGD=0, GPU=0,
              seconds=time.perf_counter()-started, ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__': main()
