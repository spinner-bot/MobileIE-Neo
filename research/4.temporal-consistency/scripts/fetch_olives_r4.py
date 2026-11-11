"""Pinned systematic subset from author's train/test lists, with verified pairs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request

import numpy as np

from exploration_r2_data import ROOT
from explore_temporal_controls import save_json

REPO = 'OLiVES-Lowlight/OLiVES'
REVISION = '291ae19d061dedcff8e62f9b6d9d33d7ceeedbaf'


def read(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=180) as stream:
                return stream.read()
        except (OSError, TimeoutError) as error:
            print(json.dumps({'url': url, 'attempt': attempt+1, 'error': str(error)}), flush=True)
            if attempt == 2:
                raise
            time.sleep(attempt+1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    selection, source_text = [], {}
    for name, count in [('train', 4), ('test', 6)]:
        url = f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{name}_list.txt'
        payload = read(url)
        source_text[name] = {'url': url, 'sha256': hashlib.sha256(payload).hexdigest(), 'text': payload.decode()}
        names = payload.decode().split()
        indices = np.linspace(0, len(names)-1, count, dtype=int).tolist()
        selection.extend({'scene': names[i], 'split': name, 'list_index': i} for i in indices)
    save_json(out/'selection.json', {'repo': REPO, 'revision': REVISION, 'author_lists': source_text,
                                    'selection': selection, 'frames': 16})

    def listing(item):
        side, scene = item
        url = f'https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/Frames/{side}/{scene}?limit=1000'
        rows = json.loads(read(url))
        return side, scene, {Path(r['path']).name: r for r in rows if r['type'] == 'file' and r['path'].endswith('.png')}

    jobs = [(side, row['scene']) for row in selection for side in ['gt', 'input']]
    with ThreadPoolExecutor(max_workers=4) as executor:
        listings = {(side, scene): rows for side, scene, rows in executor.map(listing, jobs)}
    selected = []
    for row in selection:
        gt, low = listings[('gt', row['scene'])], listings[('input', row['scene'])]
        common = sorted(set(gt) & set(low))
        if len(common) < 16:
            raise ValueError('not enough aligned names: '+row['scene'])
        for name in common[:16]:
            for side, mapping in [('gt', gt), ('input', low)]:
                selected.append({**row, 'side': side, **mapping[name]})
    save_json(out/'source_files.json', selected)

    def fetch(row):
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{row['path']}"
        path = ROOT/'.cache/data/olives-r4'/row['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            data = read(url)
            if len(data) != row['size'] or hashlib.sha256(data).hexdigest() != row['lfs']['oid']:
                raise ValueError('download integrity mismatch: '+row['path'])
            partial = path.with_suffix('.partial')
            partial.write_bytes(data)
            partial.rename(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row['lfs']['oid']:
            raise ValueError('cache integrity mismatch: '+row['path'])
        print(json.dumps({'scene': row['scene'], 'side': row['side'], 'file': path.name, 'bytes': path.stat().st_size}), flush=True)
        return {'scene': row['scene'], 'split': row['split'], 'side': row['side'], 'source_path': row['path'],
                'path': str(path.relative_to(ROOT)), 'url': url, 'sha256': digest, 'bytes': path.stat().st_size}

    with ThreadPoolExecutor(max_workers=4) as executor:
        files = list(executor.map(fetch, selected))
    assert len(files) == 320
    train = {r['sha256'] for r in files if r['split'] == 'train'}
    test = {r['sha256'] for r in files if r['split'] == 'test'}
    if train & test:
        raise ValueError('train/test exact file overlap')
    previous = json.loads((ROOT/'research/4.temporal-consistency/data/exploration-r3/olives_manifest.json').read_text())
    old = {r['sha256'] for r in previous['files']}
    save_json(out/'manifest.json', {'repo': REPO, 'revision': REVISION, 'files': files, 'selection': selection,
        'integrity': {'files': len(files), 'bytes': sum(r['bytes'] for r in files), 'train_test_hash_overlap': 0,
                      'R3_sample_exact_hash_overlap': len((train | test) & old)},
        'scope': 'systematic exploratory subset of author lists; possible related takes are not ruled out by file hashes'})


if __name__ == '__main__':
    main()
