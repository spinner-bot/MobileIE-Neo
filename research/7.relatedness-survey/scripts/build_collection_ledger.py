"""Collection-only identities and inclusion provenance; no grades or analysis queue."""
import collections,json,re
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];C=BASE/'collection'
def main():
 papers=json.loads((C/'paper-candidates.json').read_text())
 resources=[dict(temporary_key=x['temporary_key'],type='paper',title=x['title'],dois=x['dois'],arxivs=x['arxivs'],relation=x['relation'],sources=x['discovery']) for x in papers]
 resources.extend([
 dict(temporary_key='anchor-mobileie-paper',type='paper',title='MobileIE: An Extremely Lightweight and Effective ConvNet for Real-Time Image Enhancement on Mobile Devices',dois=['10.1109/ICCV51701.2025.02038'],arxivs=['2507.01838'],relation='original research anchor',sources=['receipts/official-paper-pdf.json','receipts/official-supplement-large.json','receipts/official-paper-arxiv-html.json']),
 dict(temporary_key='related-ntire2025-report',type='paper',title='NTIRE 2025 Challenge on Low Light Image Enhancement: Methods and Results',arxivs=['2510.13670'],relation='official report includes MobileIE named team method; older association independent of forward citation index',sources=['web-initial-9.json','receipts/ntire2025-paper.json','receipts/ntire2025-html.json'])])
 resources.append(dict(temporary_key='related-rgb2025-report',type='paper',title='RGB Photo Enhancement on Mobile GPUs, Mobile AI 2025 Challenge: Report',relation='official Mobile AI image-enhancement challenge report found through explicit MobileIE search; exact method/context pending Gate2',sources=['web-initial-2.json','receipts/related-rgb2025-paper.json']))
 revs=json.loads((C/'immutable-revisions.json').read_text())
 for repo in ['lhfgghc/LLVE-AMNet','lhfgghc/LLVE-AMNet-NPU','clarkren/LLVE-AMNet']:
  f=C/('plugin-head-'+repo.replace('/','--')+'.json')
  if f.exists():
   d=json.loads(json.loads(f.read_text())['result']['structuredContent']['content'])[0]
   revs.append(dict(repo=repo,sha=d['sha'],tree=d['commit']['tree']['sha'],source=f.name))
 trees={};dedups=[]
 for x in revs:
  tree=x['tree'];repo=x['repo']
  if tree in trees:
   dedups.append(dict(repo=repo,sha=x['sha'],tree=tree,disposition='duplicate-current-tree',canonical=trees[tree],evidence=x['source']));continue
  trees[tree]=repo
  resources.append(dict(temporary_key='code-'+repo.replace('/','--'),type='code',title=repo,repo=repo,revision=x['sha'],tree=tree,relation='official MobileIE project, distinct MobileIE-derived tree, or author-linked implementation of an included paper',sources=[x['source'],'receipts/sourcezip-'+repo.replace('/','--')+'.json']))
 upstream='1eac02be2a0155bc6e6b087c5d92fb11feb739bf'
 for f in (C/'raw').glob('repo-*-head.json'):
  x=json.loads(f.read_text());repo=x['html_url'].split('github.com/')[1].split('/commit/')[0]
  if repo!='AVC2-UESTC/MobileIE' and x['commit']['tree']['sha']==upstream:dedups.append(dict(repo=repo,sha=x['sha'],tree=upstream,disposition='duplicate-current-tree',canonical='AVC2-UESTC/MobileIE',evidence='raw/'+f.name))
 resources.append(dict(temporary_key='other-mobileie-explanations',type='other',title='MobileIE 中英文解读、转载与传播资料族',relation='directly explains or republishes the original MobileIE claims; sources are provenance variants, not independent scientific confirmations',sources=['receipts/chinese-hyperai.json','receipts/chinese-csdn.json','receipts/chinese-papernotes.json','receipts/chinese-eggtargaryen.json']))
 resources.append(dict(temporary_key='other-mobileie-reading-notes',type='other',title='谢袁学习主页：MobileIE 与小模型蒸馏阅读资料',relation='explicit MobileIE reading entry in a public research-notes site; deeper page to be traced',sources=['receipts/chinese-reading-notes.json']))
 resources.append(dict(temporary_key='other-mobileie-rtdetr-port',type='other',title='objectdetection_script：MobileIE MBRConv3移植至RT-DETR的公开变更说明',relation='explicit module reuse; accessible changelog is evidence, full claimed commercial code not assumed available',sources=['receipts/chinese-objectdetection.json','web-initial-5.json']))
 from collect_code_leads import CORE,PORTS,NOTES
 known_repos={r.get('repo') for r in resources}
 all_hits=json.loads((C/'code-file-discovery-ledger.json').read_text()) if (C/'code-file-discovery-ledger.json').exists() else []
 for repo in sorted(CORE|PORTS):
  if repo in known_repos:continue
  hits=[x for x in all_hits if x['repo']==repo]
  resources.append(dict(temporary_key='code-'+repo.replace('/','--'),type='code',title=repo,repo=repo,revision=hits[0]['revision'],relation='Explicit MobileIE implementation/module reuse; module-port claim requires independent source review, no fidelity/performance assumed',sources=[dict(url=x['url'],path=x['path'],revision=x['revision'],acquisition='receipts/'+x['acquisition_label']+'.json') for x in hits]))
 for repo in ['heniath/underwater-image-enhancement','TheComplianceAide/Drone_EnhancedVision','Synapse-iiserk/beautiful_ppt','dnguyennk/portfolio','38260/topconf-4-pillars']:
  hits=[x for x in all_hits if x['repo']==repo]
  resources.append(dict(temporary_key='other-'+repo.replace('/','--'),type='other',title=repo+' : MobileIE related research/application material',repo=repo,relation='Explicit MobileIE discussion in independent public research/application notes; evidence and depth not yet graded',sources=[dict(url=x['url'],path=x['path'],revision=x['revision'],acquisition='receipts/'+x['acquisition_label']+'.json') for x in hits]))
 web=[]
 for f in sorted(C.glob('web-initial-*.json')):
  x=json.loads(f.read_text())
  for block in x['result'].get('content',[]):
   if block.get('type')!='text':continue
   for entry in block['text'].split('--------------------------------------------------------------------------------'):
    first=entry.strip().split('\n')[0]
    m=re.match(r'(.+) \((https?://[^\s]+)\)$',first)
    if m:
     title,url=m.groups();existing=next((w for w in web if w['url']==url),None)
     if existing:existing['discoveries'].append(f.name)
     else:web.append(dict(title=title,url=url,discoveries=[f.name],disposition='pending-source-identity-review'))
 (C/'candidate-resources.json').write_text(json.dumps(resources,ensure_ascii=False,indent=2)+'\n')
 (C/'code-duplicate-ledger.json').write_text(json.dumps(dedups,ensure_ascii=False,indent=2)+'\n')
 (C/'web-discovery-ledger.json').write_text(json.dumps(web,ensure_ascii=False,indent=2)+'\n')
 historical=collections.Counter()
 for p in papers:
  for d in p['discovery']:
   if d['source'].startswith('research/5.'):historical[d['source']]+=1
 (C/'historical-reconciliation.json').write_text(json.dumps(dict(counts=dict(historical),resolved_into_candidates=True,source_records=[dict(temporary_key=p['temporary_key'],historical=[d for d in p['discovery'] if d['source'].startswith('research/5.')]) for p in papers if any(d['source'].startswith('research/5.') for d in p['discovery'])]),ensure_ascii=False,indent=2)+'\n')
 print('candidate resources',len(resources),dict(collections.Counter(r['type'] for r in resources)),'code duplicate snapshots',len(dedups),'web leads',len(web))
 print('Note: all entries remain collection candidates; web leads still require disposition; NO Gate1 pass.')
if __name__=='__main__':main()
