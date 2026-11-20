"""Append one frozen Gate1 identity without changing prior IDs or started order."""
import copy
import datetime
import json
from zoneinfo import ZoneInfo
from catalog_state import B, read, sha, verify_freeze, verify_closed_resource


def save(path, data):
    assert not path.exists(), 'Preserve existing artifact: ' + str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def key(row):
    p = row['preliminary']
    return (-p['score'], p['family'], p['role_order'], row['temporary_key'])


def main():
    a = B / 'collection/gate1-addenda/0002'
    assert read(a / 'gate1-review001.json')['gate1_passed']
    verify_freeze(a / 'freeze-v1.json')
    old_active = read(B / 'catalog/active.json')
    assert old_active['version'] == 2
    assert sha(B / old_active['freeze']) == old_active['freeze_sha256']
    verify_freeze(B / old_active['freeze'])
    old = read(B / old_active['index'])
    assert len(old) == 124
    old_order = read(B / old_active['order'])['order']
    events = [json.loads(s) for s in (B / 'catalog/analysis-log.jsonl').read_text().splitlines() if s]
    prefix = [r['resource_id'] for r in events if r['event'] == 'begin']
    assert prefix == old_order[:14] and len(set(prefix)) == 14
    closed = [r['resource_id'] for r in events if r['event'] == 'close']
    assert closed == prefix[:13] and events[-1]['event'] == 'begin'
    for resource_id in closed:
        verify_closed_resource(resource_id, events)
    candidates = read(a / 'new-candidates.json')['rows']
    scores = read(B / 'catalog/scoring-v3.json')['new_resource_scores']
    assert len(candidates) == len(scores) == 1
    row = copy.deepcopy(candidates[0])
    p = copy.deepcopy(scores[0])
    assert p.pop('temporary_key') == row['temporary_key']
    assert p['score'] == 10 * (p['R'] + p['N'] + p['U']) == 80
    assert p['tier'] == 'A' and 0 <= p['R'] <= 4
    assert all(0 <= p[k] <= 3 for k in ['N', 'U', 'E'])
    now = datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    row.update(resource_id='MIE-0125', preliminary=p, analysis_status='pending',
               status_is_initial_snapshot=True, indexed_at=now)
    rows = copy.deepcopy(old) + [row]
    order = prefix + [r['resource_id'] for r in sorted(
        [r for r in rows if r['resource_id'] not in prefix], key=key)]
    assert len(order) == len(set(order)) == 125
    for r in rows:
        r['analysis_order'] = order.index(r['resource_id']) + 1
    for before, after in zip(old, rows):
        assert {k:v for k,v in before.items() if k != 'analysis_order'} == {
            k:v for k,v in after.items() if k != 'analysis_order'}
    save(B / 'catalog/index-v3.json', rows)
    save(B / 'catalog/order-v3.json', dict(created_at=now, version=3,
        policy='docs/0004-初级排序与静态索引规则-26112012.md',
        extension_decision='docs/0008-二次增补索引与队列决策-26112107.md',
        gate1='collection/gate1-addenda/0002/gate1-review001.json',
        preserved_started_prefix=prefix, order=order, not_a_reading_or_approval_claim=True))
    save(B / 'catalog/id-additions-v3.json', dict(created_at=now, old_ids_unchanged=True,
        rows=[dict(resource_id=row['resource_id'], temporary_key=row['temporary_key'],
                   title=row['title'], score=p['score'], analysis_order=row['analysis_order'])]))
    resource = B / 'resources/MIE-0125'
    identity = copy.deepcopy(row)
    identity.update(canonical_url='https://arxiv.org/abs/2403.10376',
                    created_at=now, family=p['family'], catalog_version=3,
                    source_snapshot_cutoff='2026-11-17',
                    identity_verification='Collection identity/scope verified; scientific analysis pending Gate2')
    save(resource / 'identity.json', identity)
    receipts = []
    for label in ['pasta-arxiv-abs', 'pasta-arxiv-pdf-v2', 'pasta-arxiv-html-v2', 'pasta-author-home']:
        receipt = B / ('collection/receipts/g1a0002-' + label + '.json')
        r = read(receipt)
        receipts.append(dict(path=str(receipt.relative_to(B / 'collection')),
            status=r['status'], http=r.get('http_status'), complete=r.get('complete'),
            archive=r.get('archive_path'), sha256=r.get('sha256')))
    save(resource / 'sources/manifest.json', dict(resource_id=row['resource_id'],
        created_at=now, snapshot_cutoff='2026-11-17', actual_collection_dates=['2026-11-20','2026-11-21'],
        receipts=receipts, discovery_evidence=row['sources'], not_yet_analyzed=True,
        adequacy='19-page v2 acquired; official code/separate supplement/formal publication unconfirmed',
        path_conventions={'receipts[].path':'relative to workstation/collection',
            'receipts[].archive':'relative to workstation',
            'discovery_evidence[].source':'relative to workstation'}))
    protected = [B / x for x in ['catalog/index-v3.json','catalog/order-v3.json',
        'catalog/id-additions-v3.json','catalog/scoring-v3.json',
        'docs/0008-二次增补索引与队列决策-26112107.md']]
    # Resource manifests remain appendable until their own Gate2 closure.
    save(B / 'catalog/freeze-v3.json', dict(created_at=now, version=3,
        files=[dict(path=str(f.relative_to(B)), sha256=sha(f)) for f in protected]))
    print('Appended MIE-0125 at order', row['analysis_order'], '; next', order[14],
          '. Active pointer not yet changed; no begin/close event created.')


if __name__ == '__main__':
    main()
