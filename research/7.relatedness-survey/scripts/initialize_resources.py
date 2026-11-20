"""Create identity cards and source manifests only; never start or close analyses."""
import datetime,hashlib,json
from pathlib import Path
from zoneinfo import ZoneInfo
B=Path(__file__).resolve().parents[1];ROOT=B.parents[1];C=B/'collection'
def main():
 rows=json.loads((B/'catalog/index.json').read_text());available={r['temporary_key']:r for r in json.loads((C/'resource-acquisitions.json').read_text())};papers={r['temporary_key']:r for r in json.loads((C/'paper-candidates.json').read_text())};now=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
 for r in rows:
  d=B/'resources'/r['resource_id'];assert not (d/'identity.json').exists(),'Do not overwrite existing identity cards';(d/'sources').mkdir(parents=True,exist_ok=True)
  extra=papers.get(r['temporary_key'],{});authors=extra.get('authors',[]);dates={'years':extra.get('years',[]),'published_date':None,'date_verification':'pending Gate2'}
  if r['temporary_key']=='anchor-mobileie-paper':
   cross=json.loads((C/'raw/initial-crossref.json').read_text())['message'];authors=[' '.join([a.get('given',''),a.get('family','')]).strip() for a in cross.get('author',[])];dates['years']=[2025]
  if r['temporary_key']=='related-edgeaware-review':
   cross=json.loads((C/'raw/newreview-crossref.json').read_text())['message'];authors=[' '.join([a.get('given',''),a.get('family','')]).strip() for a in cross.get('author',[])]
  identity={k:r.get(k) for k in ['resource_id','temporary_key','type','title','repo','revision','tree','dois','arxivs','relation']};identity.update(created_at=now,authors=authors,dates=dates,repository_owner=(r['repo'].split('/')[0] if r.get('repo') else None),family=r['preliminary']['family'],analysis_order=r['analysis_order'],initial_preliminary=r['preliminary'],source_snapshot_cutoff='2026-11-17',identity_verification='pending Gate2; repository owner is not assumed to be the paper author')
  manifest={'resource_id':r['resource_id'],'created_at':now,'source_root':'../../.. (workstation root from this sources directory)','snapshot_cutoff':'2026-11-17','receipts':available[r['temporary_key']]['receipts'],'discovery_evidence':r['sources'],'adequacy':'Materials available for analysis; not yet read or qualified','archive_location':'workstation/archive -> owned cache namespace; original URL, date and hash in referenced receipts'}
  for x in manifest['receipts']:
   f=B/x['archive'] if x.get('archive') else None
   if f and f.exists():assert hashlib.sha256(f.read_bytes()).hexdigest()==x['sha256']
  (d/'identity.json').write_text(json.dumps(identity,ensure_ascii=False,indent=2)+'\n');(d/'sources/manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 summary='# 资源工作目录\n\n身份卡和来源清单为初整理，不代表已开始深读。永久编号及初级排序见[资源目录](../catalog/README.md)。实际进度由各资源derived/report/gate2.json及顺序日志决定，初级索引不回写成伪历史。\n\n'
 for r in rows:summary+=f"- [{r['resource_id']} {r['title']}](./{r['resource_id']}/identity.json)\n"
 (B/'resources/README.md').write_text(summary)
 print('Initialized',len(rows),'identity/source pairs; no analysis begin events')
if __name__=='__main__':main()
