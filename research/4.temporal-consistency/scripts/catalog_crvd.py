"""Seven public folder-metadata requests; no CRVD image or thumbnail requests."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import traceback
from urllib.parse import urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'research/4.temporal-consistency'
sys.path.insert(0, str(ROOT/'scripts'))
from project_assets import ASSETS, mounted_data

SOURCE = '1TZhNrsJ4tmKaU7b9b_9hkJIhwMi9Tfk3'
LIMIT = 2*1024*1024
PROTOCOL = BASE/'docs/crvd-source-qualification-protocol.md'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def listing(page):
    match = re.search(r"_DRIVE_ivd'\]\s*=\s*'((?:\\.|[^'])*)';", page)
    if not match:
        raise ValueError('No public folder metadata; do not bypass login or guess IDs')
    encoded = re.sub(r'\\x([0-9a-fA-F]{2})', r'\\u00\1', match.group(1))
    decoded = json.loads('"'+encoded.replace(r"\'", "'")+'"')
    parsed = json.loads(decoded)
    if not isinstance(parsed, list) or not isinstance(parsed[0], list):
        raise ValueError('Unexpected Drive visible-listing layout')
    entries = []
    for row in parsed[0]:
        if not isinstance(row, list) or len(row) < 4:
            raise ValueError('Unexpected folder entry')
        identifier, name, mime = row[0], row[2], row[3]
        if not all(isinstance(v, str) for v in [identifier, name, mime]):
            raise ValueError('Non-string folder identity')
        entries.append(dict(id=identifier, name=name, mime_type=mime))
    if not entries or len({r['id'] for r in entries}) != len(entries):
        raise ValueError('Empty or duplicate visible identities')
    return entries


def child(entries, name):
    found = [r for r in entries if r['name'] == name
             and r['mime_type'] == 'application/vnd.google-apps.folder']
    if len(found) != 1:
        raise ValueError(f'Expected one exact folder {name}; stop without wider traversal')
    return found[0]['id']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    if args.attempt < 1:
        raise ValueError('Positive new attempt required')
    mounted_data()
    out = BASE/f'process/external-crvd/catalog{args.attempt:03d}'
    cache = ROOT/f'.cache/datasets/CRVD/catalog{args.attempt:03d}'
    assert cache.resolve().is_relative_to(ASSETS.resolve())
    out.mkdir(parents=True, exist_ok=False)
    cache.mkdir(parents=True, exist_ok=False)
    write(out/'config.json', dict(started=datetime.now().astimezone().isoformat(),
          source_folder=SOURCE, protocol_sha256=digest(PROTOCOL),
          script_sha256=digest(Path(__file__)), max_requests=7, max_HTML_bytes=LIMIT,
          scope='Visible public metadata only; training scene1 ISO1600, no image requests/retries.'))
    started, complete, requests, records = time.perf_counter(), False, 0, []

    def get(identifier, tag):
        nonlocal requests
        assert requests < 7 and re.fullmatch(r'[A-Za-z0-9_-]+', identifier)
        requests += 1
        url = 'https://drive.google.com/drive/folders/'+identifier
        began = time.perf_counter()
        with urlopen(Request(url, headers={'User-Agent': 'MobileIE-Neo-academic-metadata/1.0'}),
                     timeout=30) as response:
            if response.status != 200 or urlparse(response.url).hostname != 'drive.google.com':
                raise ValueError('Nonpublic response or unexpected redirect; no fallback')
            if response.headers.get_content_type() != 'text/html':
                raise ValueError('Unexpected payload type; no binary read')
            content = response.read(LIMIT+1)
        if len(content) > LIMIT:
            raise ValueError('Metadata byte limit exceeded')
        path = cache/(tag+'.html')
        path.write_bytes(content)
        entries = listing(content.decode('utf-8'))
        row = dict(tag=tag, folder_id=identifier, url=url,
                   path=str(path.relative_to(ROOT)), sha256=digest(path), bytes=len(content),
                   visible_entries=entries, visible_only_not_complete_dataset=True,
                   seconds=time.perf_counter()-began)
        write(out/(tag+'.json'), row)
        records.append(row)
        print(json.dumps(dict(tag=tag, visible_entries=len(entries))), flush=True)
        return entries

    try:
        root = get(SOURCE, 'root')
        for name in ['indoor_raw_gt', 'indoor_raw_noisy']:
            branch = get(child(root, name), name)
            scene = get(child(branch, 'scene1'), name+'-scene1')
            get(child(scene, 'ISO1600'), name+'-scene1-ISO1600')
        assert requests == len(records) == 7
        write(out/'manifest.json', dict(completed=True, metadata=records,
              requests=requests, image_bytes=0, image_decodes=0,
              scope='Seven observed visible folder listings; not a complete dataset acquisition.'))
        complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(out/'runtime.json', dict(completed_normally=complete, requests=requests,
              completed_metadata_pages=len(records), new_image_bytes=0, image_decodes=0,
              CNN=0, SGD=0, GPU=0, seconds=time.perf_counter()-started,
              ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__':
    main()
