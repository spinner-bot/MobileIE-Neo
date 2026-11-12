"""Freeze all128 ordinally paired PNG members from the old four train sources."""
from datetime import datetime
import json
from pathlib import PurePosixPath

from sdsd_archive_r67 import BASE, ROOT, LISTINGS, digest, write


def main():
    catalog = BASE/'process/exploration-r67/catalog001'
    assert json.loads((catalog/'runtime.json').read_text())['completed_normally']
    split_path = LISTINGS/'official-split.json'; split = json.loads(split_path.read_text())
    assert split['commit'] == '07330bb0ac124d2d56f6e92908d650dde3f7f7c2'
    out = BASE/'data/exploration-r67-sdsd'; out.mkdir(exist_ok=False)
    selections, pairings, archives = [], [], {}
    for domain, pairs in [('indoor', ['pair1', 'pair69']), ('outdoor', ['pair2', 'pair80'])]:
        metadata = json.loads((catalog/(domain+'.json')).read_text())
        index = ROOT/metadata['index']; assert digest(index) == metadata['index_sha256']
        rows = json.loads(index.read_text())
        official_test, = [r['testing_dir'] for r in split['sources'] if r['path'] == f'options/train/train_{"in" if domain == "indoor" else "out"}_sdsd.yml']
        assert not set(pairs) & set(official_test)
        archives[domain] = dict(volumes=metadata['volumes'], catalog_sha256=digest(catalog/(domain+'.json')),
                                index_path=str(index.relative_to(ROOT)), index_sha256=digest(index))
        for pair in pairs:
            roles = {}
            for role, source_role in [('input', 'input'), ('gt', 'GT')]:
                prefix = f'{domain}/{source_role}/{pair}/'
                entries = sorted([r for r in rows if not r['directory'] and r['name'].startswith(prefix)], key=lambda r: r['name'])
                assert len(entries) >= 16 and all(PurePosixPath(r['name']).suffix.lower() == '.png' for r in entries)
                ids = [int(PurePosixPath(r['name']).stem) for r in entries]
                assert ids == sorted(ids) and all(b-a == 1 for a, b in zip(ids[:-1], ids[1:]))
                roles[role] = entries[:16]
                for ordinal, row in enumerate(entries[:16]):
                    assert row['method'] in (0, 8) and not row['flags'] & 1 and row['size'] < 16*1024*1024
                    selections.append(dict(domain=domain, pair=pair, scene=f'sdsd_{domain}_{pair}', role=role,
                        ordinal=ordinal, **row, output=f'{domain}/{pair}/{role}/{PurePosixPath(row["name"]).name}'))
                roles[role+'_all_count'] = len(entries)
            assert roles['input_all_count'] == roles['gt_all_count']
            pairings.append(dict(domain=domain, pair=pair, scene=f'sdsd_{domain}_{pair}',
                total_available_pairs=roles['input_all_count'], input_names=[r['name'] for r in roles['input']],
                gt_names=[r['name'] for r in roles['gt']], pairing='Official independently sorted arrays at same ordinal index, not equal basenames.'))
    assert len(selections) == 128 and len(pairings) == 4
    write(out/'selection.json', dict(at=datetime.now().astimezone().isoformat(), source_commit=split['commit'],
          official_split_sha256=digest(split_path), protocol_sha256=digest(BASE/'docs/exploration-r67-sdsd-fixed-acquisition-protocol.md'),
          archives=archives, pairing_table=pairings, members=selections,
          compressed_payload_bytes=sum(r['compressed_size'] for r in selections), PNG_bytes=sum(r['size'] for r in selections),
          images_decoded=0, image_members_fetched=0, selection='R17 fixed train sources, each-side first16; no test/quality-based changes.'))
    print({'members':len(selections), 'compressed_bytes':sum(r['compressed_size'] for r in selections),
           'pairing_starts':[(p['scene'], p['input_names'][0], p['gt_names'][0]) for p in pairings]}, flush=True)


if __name__ == '__main__': main()
