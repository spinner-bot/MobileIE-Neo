"""Append reviewed identities and publish a queue from manual scores; no Gate2 claims."""
import copy
import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

B = Path(__file__).resolve().parents[1]
A = B / 'collection/gate1-addenda/0001'


def read(path):
    return json.loads(path.read_text())


def save(path, data):
    assert not path.exists(), f'Preserve previous artifact: {path}'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def key(row):
    p = row['preliminary']
    return (-p['score'], p['family'], p['role_order'], row['temporary_key'])


def main():
    assert read(A / 'gate1-review001.json')['gate1_passed']
    for item in read(A / 'freeze-v1.json')['files']:
        assert hashlib.sha256((B / item['path']).read_bytes()).hexdigest() == item['sha256']
    assert not (B / 'catalog/index-v2.json').exists()
    now = datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    old = read(B / 'catalog/index.json')
    assert len(old) == 113
    scores = read(B / 'catalog/scoring-v2.json')
    candidates = read(A / 'new-candidates.json')['rows']
    acq = {r['temporary_key']:r for r in read(A / 'supplement-resource-acquisitions.json')['rows']}
    corrections = {r['temporary_key']:r for r in read(A / 'identity-corrections.json')['rows']}
    scored = {r['temporary_key']:r for r in scores['new_resource_scores']}
    assert set(scored) == {r['temporary_key'] for r in candidates}
    new = []
    for candidate in candidates:
        row = copy.deepcopy(candidate)
        if row['temporary_key'] in corrections:
            row['indexed_authors'] = row.get('authors', [])
            row['authors'] = corrections[row['temporary_key']]['authors_primary']
            row['author_metadata_source'] = corrections[row['temporary_key']]['source']
        row['preliminary'] = copy.deepcopy(scored[row['temporary_key']])
        row['preliminary'].pop('temporary_key')
        p = row['preliminary']
        assert p['score'] == 10 * (p['R'] + p['N'] + p['U'])
        assert 0 <= p['R'] <= 4 and all(0 <= p[k] <= 3 for k in ['N', 'U', 'E'])
        row.update(indexed_at=now, analysis_status='pending', status_is_initial_snapshot=True)
        new.append(row)
    new.sort(key=key)
    for number, row in enumerate(new, 114):
        row['resource_id'] = f'MIE-{number:04}'
    all_rows = copy.deepcopy(old) + new
    for row in all_rows[:113]:
        row['initial_analysis_order'] = row['analysis_order']
        row['status_is_initial_snapshot'] = True
    for change in scores['existing_resource_reassessment']:
        row = next(r for r in all_rows if r['resource_id'] == change['resource_id'])
        row['preliminary_v1'] = copy.deepcopy(row['preliminary'])
        assert row['preliminary_v1']['score'] == change['previous_score']
        row['preliminary'] = {k:v for k,v in change.items() if k not in ['resource_id', 'previous_score']}
    prefix = [f'MIE-{i:04}' for i in range(1, 5)]
    remaining = sorted([r for r in all_rows if r['resource_id'] not in prefix], key=key)
    order = prefix + [r['resource_id'] for r in remaining]
    assert len(order) == len(set(order)) == 124
    for row in all_rows:
        row['analysis_order'] = order.index(row['resource_id']) + 1
    all_rows.sort(key=lambda r:r['resource_id'])
    save(B / 'catalog/index-v2.json', all_rows)
    save(B / 'catalog/order-v2.json', {
        'created_at':now, 'version':2,
        'policy':'docs/0004-初级排序与静态索引规则-26112012.md',
        'extension_decision':'docs/0006-增补索引与队列决策-26112016.md',
        'gate1':'collection/gate1-addenda/0001/gate1-review001.json',
        'preserved_started_prefix':prefix, 'order':order,
        'not_a_reading_or_approval_claim':True,
    })
    save(B / 'catalog/id-additions-v2.json', {
        'created_at':now, 'old_ids_unchanged':True,
        'rows':[{'resource_id':r['resource_id'], 'temporary_key':r['temporary_key'],
                 'title':r['title'], 'score':r['preliminary']['score'],
                 'analysis_order':r['analysis_order']} for r in new],
    })
    for row in new:
        resource = B / 'resources' / row['resource_id']
        assert not resource.exists()
        identity = {k:copy.deepcopy(row.get(k)) for k in [
            'resource_id', 'temporary_key', 'type', 'title', 'authors', 'repo', 'revision',
            'tree', 'dois', 'arxivs', 'relation', 'publication_year', 'publication_month',
            'publication_date', 'commit_date', 'date_limit', 'metadata_limit', 'author_metadata_source']}
        identity['relation'] = row.get('relation') or row.get('relation_verification')
        identity['discovery_evidence'] = copy.deepcopy(row['sources'])
        identity['discovery_path_base'] = 'workstation root; referenced collection files retain their prefix'
        if row.get('repo'):
            canonical = 'https://github.com/' + row['repo']
        elif row.get('dois'):
            canonical = 'https://doi.org/' + row['dois'][0]
        elif row.get('arxivs'):
            canonical = 'https://arxiv.org/abs/' + row['arxivs'][0]
        else:
            first_receipt = acq[row['temporary_key']]['receipts'][0]['path']
            canonical = read(B / first_receipt)['url']
        identity['canonical_url'] = canonical
        identity.update(created_at=now, family=row['preliminary']['family'],
                        analysis_order=row['analysis_order'], catalog_version=2,
                        initial_preliminary=row['preliminary'], source_snapshot_cutoff='2026-11-17',
                        identity_verification='Primary/index association established during collection; scientific adequacy pending Gate2',
                        repository_owner=row.get('repo', '').split('/')[0] or None)
        save(resource / 'identity.json', identity)
        receipts = []
        for item in acq[row['temporary_key']]['receipts']:
            receipts.append({'path':item['path'].removeprefix('collection/'),
                             'status':item['status'], 'http':item['http_status'],
                             'complete':item['complete'], 'archive':item['archive'],
                             'sha256':item['sha256']})
        sources = [s.removeprefix('collection/') if isinstance(s, str) else s for s in row['sources']]
        save(resource / 'sources/manifest.json', {
            'resource_id':row['resource_id'], 'created_at':now, 'snapshot_cutoff':'2026-11-17',
            'actual_collection_date':'2026-11-20', 'receipts':receipts,
            'discovery_evidence':sources, 'adequacy':acq[row['temporary_key']]['availability'],
            'path_conventions':{'receipts[].path':'relative to workstation/collection',
                                'receipts[].archive':'relative to workstation',
                                'discovery_evidence strings':'relative to workstation/collection',
                                'discovery_evidence objects.source':'relative to workstation or repository as explicitly prefixed'},
            'not_yet_analyzed':True,
        })
    print('Appended 11 identities, kept old IDs, wrote v2 queue. No resource begun/closed.')


if __name__ == '__main__':
    main()
