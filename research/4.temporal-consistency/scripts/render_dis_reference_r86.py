"""All preregistered static contacts; rendering is not human visual review."""
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dis_reference_io_r86 import BASE, QUALITY, LABEL, verify, cached, lines, write, stamp, digest
from causal_reference_evaluation_r75 import metas, reference_predictions
from prepare_cache_r5 import load_one


def main():
    verify()
    assert json.loads((QUALITY/'runtime.json').read_text())['completed_normally']
    entries = {(r['seed'], r['scene']): r for r in lines(QUALITY/'prediction_manifest.jsonl')}
    old = {(r['seed'], r['scene'], r['label']): r for r in reference_predictions()}
    scenes, _ = metas(); out = BASE/'report/exploration-r86-contacts'; out.mkdir(exist_ok=False)
    manifest = []; font = ImageFont.load_default()
    for meta in scenes:
        data = load_one(meta)
        for seed in (0, 1):
            raw = cached(old[seed, meta['scene'], 'MobileIE_raw'])
            recursive = cached(old[seed, meta['scene'], 'MobileIE_recursive_self05'])
            candidate = cached(entries[seed, meta['scene']])
            selected = [0, len(raw)//2, len(raw)-1]
            canvas = Image.new('RGB', (960, 1066), 'white'); draw = ImageDraw.Draw(canvas)
            draw.text((4, 4), f"seed{seed} {meta['scene']} | t={selected} | static contacts, not flicker/MOS", fill='black', font=font)
            for row, (label, values) in enumerate([('Input', data['low']), ('GT', data['gt']),
                    ('F0 raw', raw), ('F0 + half-FB rec', recursive), ('F0 + native DIS rec', candidate)]):
                y = 26+row*208
                draw.text((4, y), label, fill='black', font=font)
                for column, t in enumerate(selected):
                    view = Image.fromarray(np.clip(np.rint(values[t]), 0, 255).astype(np.uint8))
                    view = view.resize((320, 180), Image.Resampling.LANCZOS)
                    canvas.paste(view, (column*320, y+20))
            path = out/f'seed{seed}--{meta["scene"]}.png'; canvas.save(path)
            manifest.append(dict(seed=seed, scene=meta['scene'], selected_frames=selected,
                path=str(path.relative_to(BASE.parents[1])), sha256=digest(path), shape=[1066, 960],
                actual_human_reviewed=False, scope='Fixed display-only resizing; raw scientific arrays unchanged.'))
            del raw, recursive, candidate
        del data
    assert len(manifest) == 20
    write(out/'manifest.json', dict(at=stamp(), rendered=20, actual_human_reviewed=False, images=manifest,
        scope='Actual viewing must be recorded separately; no temporal playback or MOS implied.'))
    print(json.dumps(dict(rendered=20, actual_human_reviewed=False)))


if __name__ == '__main__':
    main()
