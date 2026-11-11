"""Pinned systematic subset from author's train/test lists, with verified pairs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import urllib.error

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
            if attempt == 2 or (isinstance(error, urllib.error.HTTPError) and 400 <= error.code < 500 and error.code not in [408, 429]):
                raise
            time.sleep(attempt+1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--base-manifest')
    parser.add_argument('--extra-train', type=int, default=0)
    args = parser.parse_args()
    base = json.loads(Path(args.base_manifest).read_text()) if args.base_manifest else None
    if args.extra_train < 0 or (args.extra_train and base is None):
        parser.error('extra training scenes require a base manifest')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    available = {}
    for side in ['gt', 'input']:
        url = f'https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/Frames/{side}?limit=1000'
        directory = json.loads(read(url))
        available[side] = {Path(r['path']).name for r in directory if r['type'] == 'directory'}
    eligible = available['gt'] & available['input']
    save_json(out/'available_directories.json', {side: sorted(names) for side, names in available.items()})
    selection, source_text = [], {}
    for name, count in [('train', 4), ('test', 6)]:
        url = f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{name}_list.txt'
        payload = read(url)
        source_text[name] = {'url': url, 'sha256': hashlib.sha256(payload).hexdigest(), 'text': payload.decode()}
        names = [name for name in payload.decode().split() if name in eligible]
        if len(names) < count:
            raise ValueError('not enough accessible paired scenes: '+name)
        if base:
            original = [r['scene'] for r in base['selection'] if r['split'] == name]
            additional = []
            if name == 'train' and args.extra_train:
                remaining = [n for n in names if n not in original]
                if len(remaining) < args.extra_train:
                    raise ValueError('not enough additional training scenes')
                additional = [remaining[i] for i in np.linspace(0, len(remaining)-1, args.extra_train, dtype=int)]
            indices = [names.index(n) for n in original+additional]
        else:
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
    assert len(files) == len(selection)*32
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
