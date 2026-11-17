"""Disambiguate research name hits from Internet Explorer user-agent false positives.
Collection only: retain all discovered files and provenance before any value grading.
"""
import collections,hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];C=BASE/'collection'
CORE={'nlsidf/vnrit','nlsidf/ndk-vnrit','KrakeCZ/AIchat-Android-Local-AI-Agent-','khaihoang004/dewarp','tediyankov/reef-track','SKKUAutoLab/ETSS-07-ImageEnhance','Foadsf/neural-enhance-bench','xuhangc/Plumbing','Kazuhito00/Multimodal-Node-Editor','miaory/test','gugugu531/Hispark_SocChina_2026','wbzuo/dl-modules-toolkit','FangruLinghu/PaperCode','gogoworker/FSOD','wujunye1212/wjy_vault','pbobip/wd_bishe'}
PORTS={'FifthGentleness/RTDETRU','daidaidaidaidaidai-cat/duomotai-yolo','XuYuanDaWang/Yolo-Project','Alleluia1/RTDETR-10.20','helloshuaige520/RT-DETR','mahongwei-bot/RTDETR','mahongwei-bot/RTDETR20260307-main','tpiece/RTDETR-main','lightxiu/yolo26','linchuan232/myrtdetr','Wzhi997/YOLO','cui123000/rtdetr_indoor','pls-Kaguya/LSO-RT-DETR','Alleluia1/RTDETR-main-distill1'}
NOTES={'heniath/underwater-image-enhancement','TheComplianceAide/Drone_EnhancedVision','Synapse-iiserk/beautiful_ppt','zhaoyang97/Paper-Notes','zhaoyang97/Paper-Notes-en','zms618/paper-cards','38260/topconf-4-pillars','dnguyennk/portfolio','ZejunCao/ZejunCao.github.io'}
EXISTING={'AVC2-UESTC/MobileIE','horizon921/Computational-Photography-26Fall-MobileIE','38260/MobileIE-repro','viplab-fudan/FJNDF-Pytorch','z1069614715/objectdetection_script'}
def main():
 hits={}
 for f in sorted(C.glob('plugin-code-*.json')):
  x=json.loads(f.read_text());result=x['result']
  for r in result.get('structuredContent',{}).get('results',[]):
   url=r['url'];parts=url.split('github.com/')[1].split('/');repo='/'.join(parts[:2]);sha=parts[3];path='/'.join(parts[4:])
   if url not in hits:hits[url]=dict(repo=repo,revision=sha,path=path,url=url,discoveries=[],matches=r.get('text_matches',[]))
   hits[url]['discoveries'].append(f.name)
 rows=list(hits.values());jobs=[]
 for x in rows:
  repo=x['repo']
  if repo in CORE:x.update(disposition='new-implementation-candidate',reason='Explicit MobileIE implementation/module/weight use in source; fidelity and value not yet evaluated')
  elif repo in PORTS:x.update(disposition='module-port-claim-candidate',reason='Explicit MBRConv3/configuration reuse; publication originality, duplicate tutorial content and actual core files still to verify')
  elif repo in NOTES:x.update(disposition='interpretation-or-application-note-candidate',reason='Original public source discusses MobileIE; identity vs independent contribution still to verify')
  elif repo in EXISTING:x.update(disposition='existing-resource-additional-evidence',reason='Already included project identity')
  else:
   fragments=' '.join(m.get('fragment','') for m in x['matches'])
   ua=any(t in fragments.lower() for t in ['useragent','msie','youtube','mobileie":','mobileie/','mobileie\'','mobileie =','mobileie('])
   x.update(disposition='excluded-browser-identifier' if ua else 'bibliographic-or-search-index-evidence',reason='MobileIE keyword has browser/user-agent homonyms; otherwise daily lists, author directory or existing-paper indexing are not new implementation evidence')
  if repo in CORE|PORTS|NOTES|EXISTING:
   label='codehit-'+hashlib.sha256(x['url'].encode()).hexdigest()[:16]
   x['acquisition_label']=label
   jobs.append(dict(label=label,url=f'https://raw.githubusercontent.com/{repo}/{x["revision"]}/{x["path"]}',category=x['disposition'],repo=repo,revision=x['revision'],source_path=x['path']))
 for repo in sorted(CORE|PORTS|NOTES):
  f=C/('plugin-new-head-'+repo.replace('/','--')+'.json')
  if f.exists():
   d=json.loads(json.loads(f.read_text())['result']['structuredContent']['content'])[0];sha=d['sha']
  else:
   sha=next(x['revision'] for x in rows if x['repo']==repo)
  for path in ['README.md','LICENSE','requirements.txt']:
   label='codemeta-'+hashlib.sha256((repo+sha+path).encode()).hexdigest()[:16]
   jobs.append(dict(label=label,url=f'https://raw.githubusercontent.com/{repo}/{sha}/{path}',category='project-context-acquisition',repo=repo,revision=sha,source_path=path))
 (C/'code-file-discovery-ledger.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
 (C/'new-code-acquisition-jobs.json').write_text(json.dumps(jobs,ensure_ascii=False,indent=2)+'\n')
 print('unique matched files',len(rows),'repositories',len({x['repo'] for x in rows}),'acquisitions',len(jobs))
 print(dict(collections.Counter(x['disposition'] for x in rows)))
 print('new initial candidate repositories',len(CORE|PORTS|NOTES))
if __name__=='__main__':main()
