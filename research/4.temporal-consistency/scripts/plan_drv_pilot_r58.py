"""Select fixed RAW members from authoritative existing directories, no network/pixels."""
from datetime import datetime
import json

from dynamic_training_r34 import BASE, ROOT, digest


def main():
    split_path = BASE/'data/external-drv/official-split.json'
    split = json.loads(split_path.read_text())
    chosen = ['0001', '0072', '0138', '0202']
    assert set(chosen) <= set(split['splits']['train']['ids'])
    assert not set(chosen) & set(split['splits']['val']['ids']+split['splits']['test']['ids'])
    archives, sources, members = {}, {}, []
    for name in ['long.zip', 'short1.zip', 'short2.zip', 'short3.zip', 'short4.zip']:
        source = BASE/f'process/external-drv/archive-metadata-attempt002/{name}.json'
        meta = json.loads(source.read_text())
        index = ROOT/f'.cache/datasets/DRV/archive-index/classification001/{name}.index.json'
        archives[name] = {k: meta[k] for k in ['url', 'archive_bytes', 'etag', 'last_modified']}
        rows = json.loads(index.read_text())
        sources[str(index.relative_to(ROOT))] = digest(index)
        sources[str(source.relative_to(ROOT))] = digest(source)
        for scene in chosen:
            candidates = sorted([r for r in rows if r['matched_static_scene'] == scene
                                 and not r['directory'] and r['name'].lower().endswith('.arw')],
                                key=lambda r: r['name'])
            if not candidates: continue
            selected = candidates if name == 'long.zip' else candidates[:8]
            assert len(selected) == (1 if name == 'long.zip' else 8)
            for t, row in enumerate(selected):
                role = 'long' if name == 'long.zip' else 'short'
                members.append(dict(scene=scene, role=role, ordinal=t, archive=name,
                                    output=f'{scene}/{role}/{row["name"].split("/")[-1]}', **row))
    assert len(members) == 36
    assert all(sum(m['scene'] == s and m['role'] == role for m in members) == n
               for s in chosen for role, n in [('short', 8), ('long', 1)])
    out = BASE/'data/external-drv/raw-pilot-plan-r58.json'
    assert not out.exists()
    out.write_text(json.dumps(dict(at=datetime.now().astimezone().isoformat(), scenes=chosen,
        source_commit=split['commit'], official_split_sha256=digest(split_path),
        protocol_sha256=digest(BASE/'docs/exploration-r58-drv-static-raw-pilot-protocol.md'),
        authoritative_catalog_sha256=digest(BASE/'process/external-drv/classification001/catalog.json'),
        source_sha256=sources, archives=archives, members=members,
        compressed_payload_bytes=sum(m['compressed_size'] for m in members),
        raw_bytes=sum(m['size'] for m in members), pixels_read=0,
        selection='R52 preselected train IDs; first8 short ARW plus unique long ARW; no PNG/val/test/dynamic'),
        indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(members=len(members), compressed_bytes=sum(m['compressed_size'] for m in members))))


if __name__ == '__main__': main()
