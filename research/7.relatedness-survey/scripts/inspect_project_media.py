"""Inventory authored project media and sample document contacts; no model execution or quality test."""
import argparse
import hashlib
import io
import json
import os
import re
import zipfile
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageOps

from catalog_state import B, load_catalog, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('resource_id')
    args = parser.parse_args()
    load_catalog()
    events = [json.loads(s) for s in (B / 'catalog/analysis-log.jsonl').read_text().splitlines() if s]
    assert events[-1]['event'] == 'begin' and events[-1]['resource_id'] == args.resource_id
    resource = B / 'resources' / args.resource_id
    output = resource / 'derived/docs/media-manifest.json'
    assert not output.exists(), 'Preserve existing media inspection'
    upstream = json.loads((resource / 'derived/docs/upstream-code-manifest.json').read_text())
    receipt = json.loads((B / upstream['archive_receipt']).read_text())
    archive = B / receipt['archive_path']
    assert sha(archive) == receipt['sha256']
    cache = (B / '../../.cache/relatedness-survey/derived' / args.resource_id / 'media').resolve()
    cv2.setNumThreads(1)
    rows = []

    def save(relative, data):
        target = resource / 'derived' / relative
        destination = cache / relative
        assert not target.exists() and not target.is_symlink() and not destination.exists()
        target.parent.mkdir(parents=True, exist_ok=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        target.symlink_to(os.path.relpath(destination, target.parent))
        return target, destination

    with zipfile.ZipFile(archive) as package:
        prefix = package.namelist()[0].split('/')[0] + '/'
        for info in package.infolist():
            source = info.filename.removeprefix(prefix)
            extension = Path(source).suffix.lower()
            if info.is_dir() or extension not in ('.jpg', '.jpeg', '.png', '.mp4'):
                continue
            raw = package.read(info.filename)
            name = re.sub(r'[^a-zA-Z0-9_.-]', '_', source.replace('/', '--'))
            link, local = save('media/' + name, raw)
            row = {'source_path':source, 'bytes':len(raw), 'sha256':hashlib.sha256(raw).hexdigest(),
                   'original_copy':str(link.relative_to(resource))}
            if extension != '.mp4':
                with Image.open(io.BytesIO(raw)) as picture:
                    row.update(kind='image', width=picture.width, height=picture.height)
            else:
                capture = cv2.VideoCapture(str(local))
                assert capture.isOpened(), source
                count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = capture.get(cv2.CAP_PROP_FPS)
                width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                desired = {round(i * max(0, count-1) / 5) for i in range(6)} if count else set(range(6))
                samples = []
                decoded = 0
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if decoded in desired:
                        picture = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        samples.append((decoded, ImageOps.contain(picture, (620,350))))
                    decoded += 1
                capture.release()
                contact = Image.new('RGB', (1920,440 * max(1,(len(samples)+2)//3)), '#eeeeee')
                draw = ImageDraw.Draw(contact)
                for number, (index, picture) in enumerate(samples):
                    x,y = (number%3)*640+10,(number//3)*440+10
                    draw.text((x,y), f'{source} | source frame {index}', fill='black')
                    contact.paste(picture,(x,y+35))
                data = io.BytesIO()
                contact.save(data, format='PNG')
                preview,_ = save('figures/media-' + name + '-contact.png', data.getvalue())
                row.update(kind='video', header_frame_count=count, decoded_frame_count=decoded,
                           header_fps=fps, width=width, height=height,
                           sampled_frame_indices=[i for i,_ in samples],
                           contact=str(preview.relative_to(resource)),
                           contact_sha256=hashlib.sha256(data.getvalue()).hexdigest(),
                           full_video_watched_by_reviewer=False)
            rows.append(row)
    output.write_text(json.dumps({'resource_id':args.resource_id,
                                  'source_receipt':upstream['archive_receipt'],
                                  'source_revision':upstream['revision'],
                                  'scope':'Authored media decoding and sampled contacts only; not model execution, reproduction, video/MOS or temporal quality measurement',
                                  'rows':rows}, ensure_ascii=False, indent=2) + '\n')
    print('Inventoried', len(rows), 'authored assets; sampled contacts require actual visual inspection.')


if __name__ == '__main__':
    main()
