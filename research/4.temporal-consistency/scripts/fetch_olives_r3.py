"""Fetch only preregistered author samples; verify every file against LFS SHA256."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

from exploration_r2_data import ROOT
from explore_temporal_controls import save_json


def read_json(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    repo = 'OLiVES-Lowlight/OLiVES-sample'
    metadata = read_json('https://huggingface.co/api/datasets/'+repo)
    revision = metadata['sha']
    tree = read_json(f'https://huggingface.co/api/datasets/{repo}/tree/{revision}?recursive=true&limit=1000')
    selected = [r for r in tree if r['type'] == 'file' and any(r['path'].startswith(f'{side}/example_{scene}/')
                for side in ['gt', 'input'] for scene in [1, 2]) and r['path'].endswith('.png')
                and int(Path(r['path']).stem) < 16]
    if len(selected) != 64:
        raise ValueError(f'expected 64 paired files, found {len(selected)}')
    out = Path(args.manifest)
    save_json(out.with_name('olives_repository_metadata.json'),
              {'repo': repo, 'revision': revision, 'selected_file_metadata': selected})

    def fetch(row):
        url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{row['path']}"
        path = ROOT/'.cache/data/olives-r3'/row['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with urllib.request.urlopen(url, timeout=180) as response:
                payload = response.read()
            if len(payload) != row['size'] or hashlib.sha256(payload).hexdigest() != row['lfs']['oid']:
                raise ValueError('download hash/size mismatch: '+row['path'])
            path.write_bytes(payload)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row['lfs']['oid']:
            raise ValueError('cached image hash mismatch: '+row['path'])
        print(json.dumps({'file': row['path'], 'bytes': row['size'], 'sha256': digest}), flush=True)
        return {'path': str(path.relative_to(ROOT)), 'source_path': row['path'], 'url': url,
                'sha256': digest, 'bytes': row['size']}

    with ThreadPoolExecutor(max_workers=4) as executor:
        rows = list(executor.map(fetch, selected))
    save_json(out, {'repo': repo, 'revision': revision, 'files': rows,
        'source_project': 'https://anonymous-olives.github.io/',
        'selection': 'example_1/2 frames00000..00015 GT/input; sample3 reserved',
        'scope': 'author sample subset, not full benchmark; native registered pairs may contain residual mismatch'})


if __name__ == '__main__':
    main()
