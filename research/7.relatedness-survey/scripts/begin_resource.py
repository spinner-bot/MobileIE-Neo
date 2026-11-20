"""Enforce sequential analysis start from immutable catalog and completed prior Gates."""
import argparse,datetime,hashlib,json
from pathlib import Path
from zoneinfo import ZoneInfo
from catalog_state import load_catalog, verify_closed_resource
B=Path(__file__).resolve().parents[1];ROOT=B.parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('resource_id');a=ap.parse_args()
 active,_,order=load_catalog();assert a.resource_id in order
 log=B/'catalog/analysis-log.jsonl';events=[json.loads(s) for s in log.read_text().splitlines() if s] if log.exists() else []
 i=order.index(a.resource_id)
 for previous in order[:i]:
  verify_closed_resource(previous,events)
 if events and events[-1]['event']=='begin':
  assert events[-1]['resource_id']==a.resource_id,'Another resource still open'
  print('Existing active resource',a.resource_id,'no duplicate begin');return
 if any(e['event']=='begin' and e['resource_id']==a.resource_id for e in events):raise SystemExit('Resource already began earlier; inspect closure before reopening')
 d=B/'resources'/a.resource_id
 for q in ['derived/docs','derived/report','derived/figures']:(d/q).mkdir(parents=True,exist_ok=True)
 event=dict(event='begin',resource_id=a.resource_id,at=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),order=i+1,catalog_version=active['version'],catalog_path=active['index'],catalog_sha256=hashlib.sha256((B/active['index']).read_bytes()).hexdigest(),order_sha256=hashlib.sha256((B/active['order']).read_bytes()).hexdigest(),gate2_standard_sha256=hashlib.sha256((B/'gates/gate2-v1.md').read_bytes()).hexdigest())
 with log.open('a') as f:f.write(json.dumps(event,ensure_ascii=False)+'\n')
 print('Began',a.resource_id,'only resource allowed to be analyzed now')
if __name__=='__main__':main()
