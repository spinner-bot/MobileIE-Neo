"""Archive all full contact artifacts and rebuild legible compact display plots."""
from datetime import datetime
import json
import os
from pathlib import Path
import time
import traceback

import matplotlib
from PIL import Image, ImageDraw, ImageFont

from dynamic_training_r34 import BASE, ROOT, digest


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    render = BASE/'process/exploration-r64/render001/runtime.json'
    runtime = json.loads(render.read_text())
    assert runtime['completed_normally'] and runtime['contacts'] == 20
    folder = BASE/'report/exploration-r64-dense-history'
    source = folder/'visual_sources.json'
    old_bytes = source.read_bytes()
    old = json.loads(old_bytes)
    assert len(old['contacts']) == 20 and 'packing' not in old
    process = BASE/'process/exploration-r64/pack001'
    process.mkdir(exist_ok=False)
    full_manifest = process/'original_full_manifest.json'
    full_manifest.write_bytes(old_bytes)
    archive = ROOT/'.cache/experiments/exploration-r64/full-contact-archive001'
    assert archive.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    archive.mkdir(parents=True, exist_ok=False)
    write(process/'config.json', {'at': datetime.now().astimezone().isoformat(),
        'protocol_sha256': digest(BASE/'docs/exploration-r64-final-visual-protocol.md'),
        'pack_source_sha256': digest(Path(__file__)), 'original_manifest_sha256': digest(full_manifest),
        'render_runtime_sha256': digest(render), 'max_tile_width': 266,
        'method': 'All full-frame tiles at fixed positions; Lanczos resizing, labels redrawn14px; no exposure changes.'})
    font = ImageFont.truetype(str(Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'), 14)
    started, complete, contacts = time.perf_counter(), False, []
    try:
        for item in old['contacts']:
            path = ROOT/item['contact']
            assert digest(path) == item['sha256']
            _, height, width, channels = item['predictions'][0]['shape']
            assert channels == 3
            tile_w = min(width, 266)
            tile_h = round(height*tile_w/width)
            canvas = Image.new('RGB', (9*tile_w, 58+3*(tile_h+24)), 'white')
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 3), f"R64 {item['scene']} / seed{item['seed']} / fixed {item['frames']} / final12000 / reused DEV", font=font, fill='black')
            for col, label in enumerate(item['labels']):
                draw.text((col*tile_w+6, 29), label, font=font, fill='black')
            with Image.open(path) as original:
                assert original.size == (9*width, 58+3*(height+24))
                for row, frame in enumerate(item['frames']):
                    y = 58+row*(tile_h+24)
                    draw.text((6, y), f'Frame {frame}', font=font, fill='black')
                    for col in range(9):
                        top = 58+row*(height+24)+24
                        tile = original.crop((col*width, top, (col+1)*width, top+height))
                        if tile.size != (tile_w, tile_h):
                            tile = tile.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
                        canvas.paste(tile, (col*tile_w, y+24))
            compact = archive/('compact-'+path.name)
            full = archive/path.name
            canvas.save(compact)
            assert not full.exists()
            os.replace(path, full)
            assert digest(full) == item['sha256']
            os.replace(compact, path)
            contacts.append({**item, 'sha256': digest(path),
                'full_contact_archive': str(full.relative_to(ROOT)), 'full_contact_sha256': item['sha256'],
                'original_tile_size': [width, height], 'display_tile_size': [tile_w, tile_h],
                'display_resampling_only': True})
        assert len(contacts) == 20
        write(source, {**old, 'contacts': contacts, 'packing': {
            'at': datetime.now().astimezone().isoformat(),
            'original_full_manifest': str(full_manifest.relative_to(ROOT)),
            'original_full_manifest_sha256': digest(full_manifest),
            'archive': str(archive.relative_to(ROOT)),
            'scope': 'Full contacts preserved in owned assets; compact plots weaken fine-noise visibility. Metrics unchanged.'}})
        complete = True
    except BaseException:
        (process/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(process/'runtime.json', {'completed_normally': complete, 'contacts': len(contacts),
            'new_CNN_sequences': 0, 'new_SGD_updates': 0, 'new_GPU_jobs': 0,
            'seconds': time.perf_counter()-started, 'ended_at': datetime.now().astimezone().isoformat()})


if __name__ == '__main__':
    main()
