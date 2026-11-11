"""Fetch preregistered unused official TRAIN clips as project development data."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

from exploration_r2_data import ROOT
from explore_temporal_controls import save_json, sha256
from fetch_olives_r4 import REPO, REVISION, read


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    out = Path(p.parse_args().out)
    out.mkdir(parents=True, exist_ok=False)
    source = ROOT/'research/4.temporal-consistency/data/exploration-r4b'
    a = json.loads((source/'selection.json').read_text())
    directories = json.loads((source/'available_directories.json').read_text())
    eligible = set(directories['gt']) & set(directories['input'])
    used = {r['scene'] for r in a['selection']}
    remaining = [n for n in a['author_lists']['train']['text'].split() if n in eligible and n not in used]
    scenes = [remaining[i] for i in [0, 6, 12, 18]]
    assert len(remaining)==19 and scenes==['video_15','video_47','video_91','video_121']
    selection = [{'scene':s, 'split':'dev'} for s in scenes]
    save_json(out/'selection.json', {'repo':REPO,'revision':REVISION,'selection':selection,
        'remaining_train':remaining,'source_selection_sha256':sha256(source/'selection.json'),
        'protocol_sha256':sha256(ROOT/'research/4.temporal-consistency/docs/exploration-r9-protocol.md')})
    files = []
    for scene in scenes:
        sides = {}
        for side in ['gt','input']:
            url = f'https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/Frames/{side}/{scene}?limit=1000'
            sides[side] = {Path(r['path']).name:r for r in json.loads(read(url)) if r['type']=='file' and r['path'].endswith('.png')}
        names = sorted(set(sides['gt']) & set(sides['input']))[:16]
        assert len(names)==16
        files.extend({'scene':scene,'split':'dev','side':side,**sides[side][name]} for name in names for side in ['gt','input'])
    save_json(out/'source_files.json',files)
    def fetch(row):
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{row['path']}"
        path = ROOT/'.cache/data/olives-r4'/row['path']
        path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            payload = read(url)
            if len(payload)!=row['size'] or hashlib.sha256(payload).hexdigest()!=row['lfs']['oid']:
                raise ValueError('download mismatch '+row['path'])
            partial = path.with_suffix('.partial')
            partial.write_bytes(payload)
            partial.rename(path)
        digest = sha256(path)
        if digest!=row['lfs']['oid'] or path.stat().st_size!=row['size']:
            raise ValueError('cache mismatch '+row['path'])
        print(json.dumps({'path':row['path'],'bytes':row['size']}),flush=True)
        return {'scene':row['scene'],'split':'dev','side':row['side'],'source_path':row['path'],
                'path':str(path.relative_to(ROOT)),'sha256':digest,'bytes':row['size'],'url':url}
    with ThreadPoolExecutor(max_workers=4) as executor:
        fetched = list(executor.map(fetch,files))
    old = {}
    for name in ['exploration-r3/olives_manifest.json','exploration-r4/manifest.json','exploration-r4b/manifest.json']:
        old[name] = {r['sha256'] for r in json.loads((ROOT/'research/4.temporal-consistency/data'/name).read_text())['files']}
    hashes = {r['sha256'] for r in fetched}
    overlap = {name:len(hashes & h) for name,h in old.items()}
    if any(overlap.values()):raise ValueError('existing file overlap '+str(overlap))
    save_json(out/'manifest.json', {'repo':REPO,'revision':REVISION,'selection':selection,'files':fetched,
        'integrity':{'files':len(fetched),'bytes':sum(r['bytes'] for r in fetched),'prior_hash_overlap':overlap},
        'scope':'unused official TRAIN -> project dev; exact file disjoint, physical scene independence unverified'})


if __name__=='__main__':main()
