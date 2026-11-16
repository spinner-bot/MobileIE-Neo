"""Eight preselected train-noise contact sheets; actual viewing is separate."""
import os
if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise SystemExit('Initially empty CUDA mask required.')

import json
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from attention_mediation_r81 import ROOT, BASE, LABELS, verify
from causal_reference_io_r75 import lines, write, append, stamp, digest


def main():
    verify(); source = BASE/'results/exploration-r81-attention-noise001'
    assert json.loads((source/'runtime.json').read_text())['completed_normally']
    entries = lines(source/'prediction_manifest.jsonl'); assert len(entries) == 8
    out = BASE/'report/exploration-r81-contacts'; out.mkdir(exist_ok=False)
    font = ImageFont.load_default(size=14); started = time.perf_counter()
    for entry in entries:
        def owned(name):
            path = ROOT/entry[name]
            assert path.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
            return path
        values = np.load(owned('output_path'), mmap_mode='r')
        inputs = np.load(owned('input_path'), mmap_mode='r')
        target = np.load(owned('GT_path'))
        assert tuple(values.shape) == (5, 32, 3, 128, 128) and entry['labels'] == list(LABELS)
        assert digest(owned('output_path')) == entry['output_sha256']
        assert digest(owned('input_path')) == entry['input_sha256']
        canvas = Image.new('RGB', (1792, 554), 'white'); draw = ImageDraw.Draw(canvas)
        draw.text((8, 5), f"R81 training {entry['source_video']} / photons{entry['photons']} / seed{entry['seed']}; clean_* nondeployable; draws0/1/31", fill='black', font=font)
        for row, t in enumerate((0, 1, 31)):
            views = [('Input', inputs[t]), ('GT', target)]
            views += [(label+'(clean)' if label.startswith('clean_') else label,
                       values[i, t].transpose(1, 2, 0)) for i, label in enumerate(LABELS)]
            for column, (label, array) in enumerate(views):
                x, y = column*256, 36+row*172
                draw.text((x+2, y), f'{label} n={t}', fill='black', font=font)
                frame = Image.fromarray(np.clip(array*255, 0, 255).astype(np.uint8))
                canvas.paste(frame, (x+64, y+23))
        name = f"seed{entry['seed']}--source{entry['source_ordinal']}--ph{entry['photons']}.png"
        path = out/name; canvas.save(path)
        append(out/'manifest.jsonl', dict(path=str(path.relative_to(ROOT)), sha256=digest(path),
            seed=entry['seed'], source_video=entry['source_video'], source_ordinal=entry['source_ordinal'],
            photons=entry['photons'], draws=[0, 1, 31], actual_reviewed=False))
    write(out/'runtime.json', dict(ended=stamp(), completed_normally=True, contacts=8,
        actual_reviewed=False, seconds=time.perf_counter()-started, new_CNN=0, new_SGD=0,
        scope='Generation only, static128px training crop views, not real video playback or MOS.'))


if __name__ == '__main__':
    main()
