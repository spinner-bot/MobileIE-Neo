"""Classify already acquired raw ZIP directories by parent scene IDs only."""
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import PurePosixPath

from dynamic_training_r34 import BASE, ROOT, digest


def main():
    source = BASE/'process/external-drv/archive-metadata-attempt002'
    assert json.loads((source/'runtime.json').read_text())['completed_normally']
    split = json.loads((BASE/'data/external-drv/official-split.json').read_text())
    train = sorted(split['splits']['train']['ids'])
    static_ids = {v for row in split['splits'].values() for v in row['ids']}
    chosen = [train[i] for i in [0, len(train)//3, 2*len(train)//3, len(train)-1]]
    out = BASE/'process/external-drv/classification001'; out.mkdir(exist_ok=False)
    cache = ROOT/'.cache/datasets/DRV/archive-index/classification001'; cache.mkdir(exist_ok=False)
    reports = []
    for p in sorted(source.glob('*.zip.json')):
        record = json.loads(p.read_text()); index = ROOT/record['index_path']
        assert digest(index) == record['index_sha256']
        values = json.loads(index.read_text()); changes = 0
        counts, dynamic_counts = Counter(), Counter()
        for row in values:
            parents = PurePosixPath(row['name']).parts[:-1]
            dynamic = [part for part in parents if re.fullmatch(r'M\d{4}', part)]
            static = [part for part in parents if part in static_ids]
            assert not (dynamic and static)
            scene = static[-1] if static else None
            changes += scene != row['matched_static_scene']
            row['matched_static_scene'] = scene
            row['dynamic_scene'] = dynamic[-1] if dynamic else None
            if not row['directory']:
                if scene: counts[scene] += 1
                if dynamic: dynamic_counts[dynamic[-1]] += 1
        target = cache/index.name; target.write_text(json.dumps(values, indent=2)+'\n')
        reports.append({'archive': p.name.removesuffix('.json'), 'source_metadata': str(p.relative_to(ROOT)),
            'source_metadata_sha256': digest(p), 'source_index_sha256': record['index_sha256'],
            'corrected_index': str(target.relative_to(ROOT)), 'corrected_index_sha256': digest(target),
            'corrected_static_labels': changes, 'static_scene_counts': dict(sorted(counts.items())),
            'dynamic_scene_counts': dict(sorted(dynamic_counts.items())),
            'candidate_raw_members': {scene: [r for r in values if r['matched_static_scene'] == scene
                and PurePosixPath(r['name']).suffix.lower() == '.arw' and r['size'] > 0] for scene in chosen}})
    shorts = [r for r in reports if r['archive'].startswith('short')]
    long = next(r for r in reports if r['archive'] == 'long.zip')
    assert {k for r in shorts for k in r['static_scene_counts']} == static_ids
    assert set(long['static_scene_counts']) == static_ids
    result = {'at': datetime.now().astimezone().isoformat(), 'processing_complete': True,
        'archives': reports, 'future_train_candidates': chosen, 'all_202_static_scenes_present_in_raw_and_long': True,
        'source_bytes_refetched': 0, 'image_members_opened': 0, 'video_frames_decoded': 0,
        'scope': 'Correct local metadata labels: dynamic Mxxxx/frame0001_*.ARW is not static scene0001. '
                 'Original indexes and metadata attempt002 retained, labels there superseded; no source/dataset change.'}
    (out/'catalog.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'static_raw': len(static_ids), 'dynamic_raw': len({k for r in shorts for k in r['dynamic_scene_counts']}),
        'label_corrections': {r['archive']: r['corrected_static_labels'] for r in reports}}))


if __name__ == '__main__': main()
