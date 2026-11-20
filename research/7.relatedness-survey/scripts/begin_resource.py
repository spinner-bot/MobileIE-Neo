"""Enforce sequential analysis start from immutable catalog and completed prior Gates."""
import argparse,datetime,hashlib,json
from pathlib import Path
from zoneinfo import ZoneInfo
B=Path(__file__).resolve().parents[1];ROOT=B.parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('resource_id');a=ap.parse_args()
 assert json.loads((B/'collection/gate1-review001.json').read_text())['gate1_passed']
 for name in ['gates/freeze-v1.json','catalog/freeze-v1.json']:
  for r in json.loads((B/name).read_text())['files']:
   f=ROOT/r['path'] if name.startswith('gates/') else B/r['path']
   if not f.exists():f=B/r['path']
   assert hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256'],str(f)
 order=json.loads((B/'catalog/order-v1.json').read_text())['order'];assert a.resource_id in order
 log=B/'catalog/analysis-log.jsonl';events=[json.loads(s) for s in log.read_text().splitlines() if s] if log.exists() else []
 i=order.index(a.resource_id)
 for previous in order[:i]:
  f=B/'resources'/previous/'derived/report/gate2.json';assert f.exists(),f'Previous Gate2 missing: {previous}'
  g=json.loads(f.read_text());assert g.get('review_passed') and g.get('status') in ['pass_full','closed_limited'],previous
  closes=[e for e in events if e['event']=='close' and e['resource_id']==previous];assert len(closes)==1,previous
  close=closes[0];assert hashlib.sha256(f.read_bytes()).hexdigest()==close['gate2_sha256'],previous+' Gate changed after close'
  integrity=f.parent/'integrity.json';assert hashlib.sha256(integrity.read_bytes()).hexdigest()==close['integrity_sha256'],previous+' audit changed'
  for item in json.loads(integrity.read_text())['files']:
   asset=B/'resources'/previous/item['path'];assert asset.is_file() and hashlib.sha256(asset.read_bytes()).hexdigest()==item['sha256'],str(asset)
 if events and events[-1]['event']=='begin':
  assert events[-1]['resource_id']==a.resource_id,'Another resource still open'
  print('Existing active resource',a.resource_id,'no duplicate begin');return
 if any(e['event']=='begin' and e['resource_id']==a.resource_id for e in events):raise SystemExit('Resource already began earlier; inspect closure before reopening')
 d=B/'resources'/a.resource_id
 for q in ['derived/docs','derived/report','derived/figures']:(d/q).mkdir(parents=True,exist_ok=True)
 event=dict(event='begin',resource_id=a.resource_id,at=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),order=i+1,catalog_sha256=hashlib.sha256((B/'catalog/index.json').read_bytes()).hexdigest(),gate2_standard_sha256=hashlib.sha256((B/'gates/gate2-v1.md').read_bytes()).hexdigest())
 with log.open('a') as f:f.write(json.dumps(event,ensure_ascii=False)+'\n')
 print('Began',a.resource_id,'only resource allowed to be analyzed now')
if __name__=='__main__':main()
