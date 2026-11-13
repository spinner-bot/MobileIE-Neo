"""Render the four prescribed source contacts only after complete R71 acquisition."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'research/4.temporal-consistency'


def main():
    fetch = BASE / 'process/exploration-r71/fetch001'
    runtime = json.loads((fetch / 'runtime.json').read_text())
    assert runtime['completed_normally'] and runtime['completed_members'] == 256
    assert runtime['validated_png_files_on_disk'] == 256
    rows = [json.loads(x) for x in (fetch / 'manifest.jsonl').read_text().splitlines()]
    assert len(rows) == len({r['member'] for r in rows}) == 256
    index = {(Path(r['member']).parts[-2], int(Path(r['member']).stem)): r for r in rows}
    assert set(index) == {(s, t) for s in ['000', '009', '019', '029'] for t in range(64)}
    out = BASE / 'report/exploration-r71-reds-sources'
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    sources = []
    for scene in ['000', '009', '019', '029']:
        contact = Image.new('RGB', (1440, 325), 'white')
        draw = ImageDraw.Draw(contact)
        draw.text((8, 5), f'REDS val sharp {scene} / fixed DEV / input source only / full FOV', fill='black')
        for column, frame in enumerate([0, 32, 63]):
            row = index[scene, frame]
            source = ROOT / row['path']
            assert hashlib.sha256(source.read_bytes()).hexdigest() == row['sha256']
            with Image.open(source) as image:
                assert image.mode == 'RGB' and image.size == (1280, 720)
                contact.paste(image.resize((480, 270), Image.Resampling.LANCZOS), (column * 480, 55))
            draw.text((column * 480 + 8, 32), f'Frame {frame} / RGB uint8 / native720p', fill='black')
            sources.append(dict(scene=scene, frame=frame, source=row['path'], sha256=row['sha256']))
        contact.save(out / f'fixed-{scene}.png')
    result = dict(completed_normally=True, ended=datetime.now().astimezone().isoformat(),
                  seconds=time.perf_counter() - started, contacts=4, actual_reviewed=False,
                  source_frames=12, new_CNN=0, new_SGD=0, new_GPU=0,
                  scope='Fixed source contacts only; no enhancement/degradation, color correction, scene selection or temporal quality claim.')
    (out / 'sources.json').write_text(json.dumps(sources, indent=2) + '\n')
    (out / 'runtime.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
