"""Validate immutable catalog versions, source freezes, and required appended reviews."""
import hashlib
import json
from pathlib import Path

B = Path(__file__).resolve().parents[1]
ROOT = B.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def verify_freeze(path):
    for item in read(path)['files']:
        source = ROOT / item['path'] if item['path'].startswith('research/') else B / item['path']
        assert source.is_file() and sha(source) == item['sha256'], str(source)


def verify_required_review(requirement):
    directory = B / requirement['review_dir']
    gate = directory / 'report/gate2.json'
    integrity = directory / 'report/integrity.json'
    g = read(gate)
    assert g['resource_id'] == requirement['resource_id']
    assert g['review_passed'] and g['status'] in ('pass_full', 'closed_limited')
    assert g['standard_sha256'] == sha(B / 'gates/gate2-v1.md')
    assert all(g['checklist'][k]['passed'] and g['checklist'][k]['evidence'] for k in 'abcdefg')
    log = B / 'catalog/review-log.jsonl'
    events = [json.loads(s) for s in log.read_text().splitlines() if s]
    closes = [r for r in events if r['event'] == 'close_review'
              and r['review_dir'] == requirement['review_dir']]
    assert len(closes) == 1
    assert closes[0]['gate2_sha256'] == sha(gate)
    assert closes[0]['integrity_sha256'] == sha(integrity)
    for item in read(integrity)['files']:
        source = directory / item['path']
        assert source.is_file() and sha(source) == item['sha256'], str(source)


def load_catalog(require_advancement=True):
    assert read(B / 'collection/gate1-review001.json')['gate1_passed']
    required_count = len(read(B / 'catalog/index.json'))
    for name in ['gates/freeze-v1.json', 'catalog/freeze-v1.json',
                 'collection/gate1-freeze001.json']:
        verify_freeze(B / name)
    for status_path in sorted((B / 'collection/gate1-addenda').glob('*/status.json')):
        status = read(status_path)
        freeze = status_path.parent / 'freeze-v1.json'
        if freeze.exists():
            verify_freeze(freeze)
            review = read(status_path.parent / 'gate1-review001.json')
            assert review['gate1_passed']
            required_count = max(required_count, review['total_candidate_count'])
        if require_advancement:
            assert status.get('advance_allowed') is True, 'Unresolved Gate1 addendum/re-review: ' + str(status_path)
            for requirement in status.get('required_reviews', []):
                verify_required_review(requirement)
    active_path = B / 'catalog/active.json'
    if active_path.exists():
        active = read(active_path)
        assert sha(B / active['freeze']) == active['freeze_sha256']
        verify_freeze(B / active['freeze'])
        protected = {r['path'] for r in read(B / active['freeze'])['files']}
        assert active['index'] in protected and active['order'] in protected
        assert sha(B / active['index']) == active['index_sha256']
        assert sha(B / active['order']) == active['order_sha256']
    else:
        active = {'version':1, 'index':'catalog/index.json',
                  'order':'catalog/order-v1.json', 'freeze':'catalog/freeze-v1.json'}
    index = read(B / active['index'])
    order = read(B / active['order'])['order']
    assert len(order) == len(set(order)) == len(index)
    assert len(index) >= required_count, 'Catalog predates passed supplemental collection'
    assert set(order) == {r['resource_id'] for r in index}
    return active, index, order


def verify_closed_resource(resource_id, events):
    resource = B / 'resources' / resource_id
    gate = resource / 'derived/report/gate2.json'
    integrity = resource / 'derived/report/integrity.json'
    g = read(gate)
    assert g.get('review_passed') and g.get('status') in ('pass_full', 'closed_limited')
    closes = [r for r in events if r['event'] == 'close' and r['resource_id'] == resource_id]
    assert len(closes) == 1
    assert sha(gate) == closes[0]['gate2_sha256'], resource_id + ' Gate changed'
    assert sha(integrity) == closes[0]['integrity_sha256'], resource_id + ' audit changed'
    for item in read(integrity)['files']:
        source = resource / item['path']
        assert source.is_file() and sha(source) == item['sha256'], str(source)
