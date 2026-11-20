"""Attach collected source versions to unclosed resources; preserve closed resource audits."""
import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

B = Path(__file__).resolve().parents[1]
A = B / 'collection/gate1-addenda/0001'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = B / 'catalog/source-additions-v2.json'
    assert not out.exists(), 'Preserve prior intake application'
    assert json.loads((A / 'gate1-review001.json').read_text())['gate1_passed']
    rows = json.loads((A / 'existing-resource-source-additions.json').read_text())['rows']
    rows.append({'resource_id':'MIE-0059',
                 'reason':'New official author project explicitly acknowledges MobileIE; companion source for paper identity/priority, not substitute for unavailable fulltext',
                 'receipt_labels':['g1a0001-ulightif-meta', 'g1a0001-ulightif-head',
                                   'g1a0001-ulightif-sourcezip']})
    actions = []
    for row in rows:
        resource = B / 'resources' / row['resource_id']
        if (resource / 'derived/report/integrity.json').exists():
            assert row['resource_id'] == 'MIE-0003'
            actions.append(dict(row, disposition='Deferred to separate formal-version re-review; old manifest untouched'))
            continue
        path = resource / 'sources/manifest.json'
        before = sha(path)
        manifest = json.loads(path.read_text())
        present = {r['path'] for r in manifest['receipts']}
        added = []
        for label in row['receipt_labels']:
            relative = 'receipts/' + label + '.json'
            assert relative not in present
            receipt = json.loads((B / 'collection' / relative).read_text())
            manifest['receipts'].append({
                'path':relative, 'status':receipt['status'], 'http':receipt.get('http_status'),
                'complete':receipt.get('complete'), 'archive':receipt.get('archive_path'),
                'sha256':receipt.get('sha256')})
            added.append(relative)
        manifest.setdefault('source_additions', []).append({
            'catalog_version':2, 'gate1':'collection/gate1-addenda/0001/gate1-review001.json',
            'reason':row['reason'], 'added':added})
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        actions.append(dict(row, disposition='Applied to unclosed resource manifest',
                            manifest=str(path.relative_to(B)), before_sha256=before,
                            after_sha256=sha(path), added=added))
    out.write_text(json.dumps({'at':datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                              'actions':actions, 'closed_original_files_changed':False},
                             ensure_ascii=False, indent=2) + '\n')
    print('Attached sources to unclosed resources; MIE-0003 old audit preserved.')


if __name__ == '__main__':
    main()
