"""Record integrity for a manually completed required re-review; never approve its content."""
import argparse
import datetime
import json
import re
from pathlib import Path
from urllib.parse import unquote
from zoneinfo import ZoneInfo

from catalog_state import B, load_catalog, sha, verify_closed_resource


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('resource_id')
    parser.add_argument('review_number')
    args = parser.parse_args()
    load_catalog(require_advancement=False)
    relative = f'resources/{args.resource_id}/derived/reviews/{args.review_number}'
    eligible = []
    for p in (B / 'collection/gate1-addenda').glob('*/status.json'):
        status = json.loads(p.read_text())
        if status.get('gate1_passed') and not status.get('advance_allowed'):
            eligible.extend(status.get('required_reviews', []))
    assert any(r['resource_id'] == args.resource_id and r['review_dir'] == relative for r in eligible)
    original_events = [json.loads(s) for s in (B / 'catalog/analysis-log.jsonl').read_text().splitlines() if s]
    verify_closed_resource(args.resource_id, original_events)
    directory = B / relative
    gate = directory / 'report/gate2.json'
    integrity_path = directory / 'report/integrity.json'
    assert not integrity_path.exists(), 'Preserve closed review audit'
    g = json.loads(gate.read_text())
    assert g['resource_id'] == args.resource_id and g['review_id'] == args.review_number
    assert g['review_passed'] and g['status'] in ('pass_full', 'closed_limited')
    assert g['standard_sha256'] == sha(B / 'gates/gate2-v1.md')
    for letter in 'abcdefg':
        check = g['checklist'][letter]
        assert check['passed'] and check['evidence']
        assert all((directory / name).is_file() for name in check['evidence'])
    if g['status'] == 'closed_limited':
        assert all(g['limited_closure'].get(k) for k in ['attempts', 'missing', 'unlock_conditions'])
    claims = json.loads((directory / 'report/claims.json').read_text())
    assert claims['resource_id'] == args.resource_id and claims['claims']
    for claim in claims['claims']:
        assert claim['evidence_class'] and claim['limitation'] and claim['anchors']
        assert all((B / anchor['source']).is_file() for anchor in claim['anchors'])
    manifest = json.loads((directory / 'sources/manifest.json').read_text())
    for item in manifest['receipts']:
        assert (B / 'collection' / item['path']).is_file()
        if item.get('archive'):
            assert sha(B / item['archive']) == item['sha256']
    integrity = []
    link_count = 0
    for path in sorted(directory.rglob('*')):
        if path.is_dir() or path.name in ('gate2.json', 'integrity.json'):
            continue
        assert path.is_file(), str(path)
        if path.suffix == '.md':
            for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', path.read_text()):
                if '://' in target or target.startswith('#'):
                    continue
                target = unquote(target.strip('<>').split('#', 1)[0])
                assert (path.parent / target).exists(), f'{path}: {target}'
                link_count += 1
        integrity.append({'path':str(path.relative_to(directory)), 'sha256':sha(path),
                          'symlink_target':str(path.readlink()) if path.is_symlink() else None})
    now = datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    integrity_path.write_text(json.dumps({
        'resource_id':args.resource_id, 'review_id':args.review_number, 'verified_at':now,
        'path_base':'review directory', 'meaning':'File/source integrity, not automatic scientific approval',
        'local_markdown_links_verified':link_count, 'original_closed_audit_verified':True,
        'files':integrity,
    }, ensure_ascii=False, indent=2) + '\n')
    log = B / 'catalog/review-log.jsonl'
    events = [json.loads(s) for s in log.read_text().splitlines() if s] if log.exists() else []
    assert not any(e['review_dir'] == relative for e in events)
    event = {'event':'close_review', 'resource_id':args.resource_id, 'review_id':args.review_number,
             'review_dir':relative, 'at':now, 'status':g['status'],
             'gate2_sha256':sha(gate), 'integrity_sha256':sha(integrity_path),
             'standard_sha256':g['standard_sha256']}
    with log.open('a') as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + '\n')
    print('Closed appended review', args.resource_id, args.review_number, g['status'],
          '; original analysis log/audit unchanged, advancement still requires hold release.')


if __name__ == '__main__':
    main()
