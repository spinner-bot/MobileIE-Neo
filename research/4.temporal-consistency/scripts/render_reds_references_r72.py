"""All16 fixed native REDS contacts from normal cached outputs, no new CNN."""
import json
import time
import numpy as np
from PIL import Image, ImageDraw

from reds_references_r72 import BASE, ROOT, lines, load_arrays, load_data_index, stamp, write


def main():
    index = load_data_index(); entries = []
    for seed in [0, 1]:
        folder = BASE / f'results/exploration-r72-quality-seed{seed}'
        terminal = json.loads((folder / 'runtime.json').read_text()); assert terminal['completed_normally']
        entries += lines(folder / 'prediction_manifest.jsonl')
    lookup = {(r['scene'], r['seed'], r['family'], r['method']): r for r in entries}
    assert len(lookup) == 144
    out = BASE / 'report/exploration-r72-reds-contacts'; out.mkdir(parents=True, exist_ok=False)
    columns = [('MobileIE_MSE', 'raw'), ('MobileIE_MSE', 'recursive_self05'),
               ('MobileIE_L1', 'raw'), ('MobileIE_L1', 'recursive_self05'),
               ('Nano', 'raw'), ('Nano', 'recursive_self05'),
               ('StableLLVE_L1', 'raw'), ('StableLLVE_L1', 'recursive_self05'), ('FastDVDnet', 'window')]
    started = time.perf_counter(); sources = []; count = 0
    for row in index['scenes']:
        data = load_arrays(row)
        for seed in [0, 1]:
            selected = [lookup[row['scene'], seed, family, method] for family, method in columns]
            arrays = [data['low'], data['gt']] + [np.load(ROOT / r['path'], mmap_mode='r', allow_pickle=False) for r in selected]
            labels = ['Input', 'GT'] + [f'{family} {"rec .5" if method == "recursive_self05" else method}' for family, method in columns]
            sheet = Image.new('RGB', (11 * 224, 940), 'white'); draw = ImageDraw.Draw(sheet)
            draw.text((8, 4), f'R72 {row["scene"]} / seed{seed} / source24fps / fixed DEV / frames0,32,63 / native-center128', fill='black')
            for col, (label, array) in enumerate(zip(labels, arrays)):
                draw.text((col * 224 + 4, 24), label, fill='black')
                for ordinal, frame in enumerate([0, 32, 63]):
                    value = np.clip(np.rint(array[frame]), 0, 255).astype(np.uint8)
                    image = Image.fromarray(value)
                    y = 50 + ordinal * 296
                    draw.text((col * 224 + 4, y), f'Frame{frame} full FOV', fill='black')
                    sheet.paste(image.resize((224, 126), Image.Resampling.LANCZOS), (col * 224, y + 15))
                    draw.text((col * 224 + 4, y + 143), 'Native center128', fill='black')
                    sheet.paste(image.crop((576, 296, 704, 424)), (col * 224 + 48, y + 158))
            path = out / f'fixed-{row["scene"]}-seed{seed}.png'; sheet.save(path)
            sources.append(dict(path=str(path.relative_to(ROOT)), scene=row['scene'], seed=seed,
                                frames=[0, 32, 63], sources=selected, actual_reviewed=False))
            count += 1
            del arrays, sheet
        del data
    assert count == 16
    write(out / 'sources.json', sources)
    write(out / 'runtime.json', dict(completed_normally=True, contacts=count, new_CNN=0, new_SGD=0,
          new_GPU=0, actual_reviewed=False, seconds=time.perf_counter() - started, ended=stamp()))


if __name__ == '__main__':
    main()
