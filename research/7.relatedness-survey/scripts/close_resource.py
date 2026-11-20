"""Verify a manually reviewed resource and append its closure; never write analysis claims."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo
from catalog_state import load_catalog, verify_closed_resource

B = Path(__file__).resolve().parents[1]
ROOT = B.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('resource_id')
    args = ap.parse_args()
    active, _, order = load_catalog()
    r = B / 'resources' / args.resource_id
    decision = r / 'derived/report/gate2.json'
    g = json.loads(decision.read_text())
    assert g['resource_id'] == args.resource_id
    assert g['review_passed'] is True
    assert g['status'] in ('pass_full', 'closed_limited')
    assert g['standard_sha256'] == sha(B / 'gates/gate2-v1.md')
    assert all(g['checklist'][k]['passed'] is True for k in 'abcdefg')
    assert all(g['checklist'][k].get('evidence') for k in 'abcdefg')
    if g['status'] == 'closed_limited':
        assert g.get('limited_closure') and g['limited_closure'].get('attempts')
        assert g['limited_closure'].get('missing') and g['limited_closure'].get('unlock_conditions')
    index = order.index(args.resource_id)
    log = B / 'catalog/analysis-log.jsonl'
    events = [json.loads(line) for line in log.read_text().splitlines() if line]
    for prior in order[:index]:
        verify_closed_resource(prior, events)
    assert events[-1]['event'] == 'begin' and events[-1]['resource_id'] == args.resource_id
    required = ['identity.json', 'sources/manifest.json', 'derived/docs/reading-notes.md',
                'derived/report/analysis.md', 'derived/report/claims.json', 'derived/docs/gate2-review.md']
    for name in required:
        assert (r / name).is_file() and (r / name).stat().st_size > 0, name
    claims = json.loads((r / 'derived/report/claims.json').read_text())
    assert claims['resource_id'] == args.resource_id and claims['claims']
    assert all(c.get('anchors') and c.get('evidence_class') and c.get('limitation') for c in claims['claims'])
    sources = json.loads((r / 'sources/manifest.json').read_text())
    for item in sources['receipts']:
        receipt = B / 'collection' / item['path']
        assert receipt.is_file(), str(receipt)
        if item.get('archive'):
            original = B / item['archive']
            assert original.is_file() and sha(original) == item['sha256'], str(original)
    upstream = r / 'derived/docs/upstream-code-manifest.json'
    if upstream.exists():
        content = json.loads(upstream.read_text())
        receipt = json.loads((B / content['archive_receipt']).read_text())
        assert sha(B / receipt['archive_path']) == receipt['sha256']
        for item in content['files']:
            assert sha(r / item['readable_path']) == item['sha256'], item['source_path']
    integrity = []
    for f in sorted(r.rglob('*')):
        if f.name in ('gate2.json', 'integrity.json') or f.is_dir():
            continue
        assert f.is_file(), f'Unresolved file or symlink: {f}'
        integrity.append({'path':str(f.relative_to(r)), 'sha256':sha(f),
                          'symlink_target':str(f.readlink()) if f.is_symlink() else None})
    check = r / 'derived/report/integrity.json'
    assert not check.exists(), 'Preserve old audit; do not silently overwrite'
    now = datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    check.write_text(json.dumps({'resource_id':args.resource_id, 'verified_at':now,
                                'meaning':'File/source integrity only; content decision remains manual self-review',
                                'files':integrity}, ensure_ascii=False, indent=2) + '\n')
    event = {'event':'close', 'resource_id':args.resource_id, 'at':now, 'order':index+1,
             'catalog_version':active['version'], 'catalog_path':active['index'],
             'catalog_sha256':sha(B / active['index']), 'order_sha256':sha(B / active['order']),
             'begin_catalog_sha256':events[-1]['catalog_sha256'],
             'status':g['status'], 'gate2_sha256':sha(decision), 'integrity_sha256':sha(check),
             'standard_sha256':g['standard_sha256']}
    with log.open('a') as f:
        f.write(json.dumps(event, ensure_ascii=False) + '\n')
    print('Closed', args.resource_id, g['status'], '; integrity verified, no experiments executed')


if __name__ == '__main__':
    main()
