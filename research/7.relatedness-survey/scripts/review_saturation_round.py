"""Audit one supplemental round. Unknown scientific identities keep the round pending."""
import argparse,datetime,hashlib,json,re
from pathlib import Path
from zoneinfo import ZoneInfo
BASE=Path(__file__).resolve().parents[1];C=BASE/'collection'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--round',type=int,required=True);ap.add_argument('--academic',required=True);args=ap.parse_args();n=args.round
 resources=json.loads((C/'candidate-resources.json').read_text());repos={r.get('repo'):r['temporary_key'] for r in resources if r.get('repo')}
 web_file=C/f'web-supplement-round{n}.json';code_file=C/f'plugin-code-supplement-round{n}.json'
 web=json.loads(web_file.read_text());code=json.loads(code_file.read_text());prior_urls=set()
 for f in list(C.glob('web-initial-*.json'))+list(C.glob('web-supplement-round*.json'))+[C/'web-new-review-trace.json']:
  if f==web_file:continue
  if 'round' in f.stem and int(re.search(r'round(\d+)',f.stem)[1])>=n:continue
  if not f.exists():continue
  for b in json.loads(f.read_text())['result'].get('content',[]):
   if b.get('type')=='text':
    for e in b['text'].split('--------------------------------------------------------------------------------'):
     m=re.match(r'(.+) \((https?://[^\s]+)\)$',e.strip().split('\n')[0])
     if m:prior_urls.add(m[2])
 web_rows=[];pending=[]
 for block in web['result'].get('content',[]):
  if block.get('type')!='text':continue
  for entry in block['text'].split('--------------------------------------------------------------------------------'):
   m=re.match(r'(.+) \((https?://[^\s]+)\)$',entry.strip().split('\n')[0])
   if not m:continue
   title,url=m.groups()
   if url in prior_urls:disposition='previously reviewed source/identity'
   elif 'MobileIE' in title and ('Extremely Lightweight' in title or '/YanLZLSZZ25/' in url):disposition='same original-paper bibliographic/translation alias'
   elif url.startswith('https://github.com/AVC2-UESTC/MobileIE/'):disposition='same original repository subresource; branches/main, no releases or PRs verified separately'
   elif any(t in title for t in ['Hailong Yan','Ce Zhu','Ce ZHU','Le Zhang','Zenglin Shi','Ao Li']):disposition='author directory/profile, not a new technical artifact'
   else:disposition='pending';pending.append(dict(kind='web',title=title,url=url))
   web_rows.append(dict(title=title,url=url,disposition=disposition))
 code_rows=[]
 for i,x in enumerate(code['result'].get('structuredContent',{}).get('results',[])):
  repo='/'.join(x['url'].split('github.com/')[1].split('/')[:2]);fragments=' '.join(m.get('fragment','') for m in x.get('text_matches',[]));path=x.get('path','')
  if repo in repos:disposition='known included resource';resource=repos[repo]
  elif re.search(r'isMobileIE|mobileIe|Windows Phone|iemobile|user.?agent|addtohome|_ua|MSIE|youtube|yt_dlp|youtube_dl',fragments+' '+path,re.I):disposition='browser/user-agent/download-tool homonym';resource=None
  elif repo.startswith('cirosantilli/expired-domain'):disposition='expired-domain numeric/name substring';resource=None
  else:disposition='pending';resource=None;pending.append(dict(kind='code',repo=repo,path=path,fragments=fragments))
  code_rows.append(dict(position=i,repo=repo,path=path,url=x['url'],disposition=disposition,resource=resource))
 a=json.loads((C/'receipts'/(args.academic+'.json')).read_text());academic=dict(receipt='receipts/'+args.academic+'.json',transport=a['status'],new=[])
 if a['status']=='success' and a.get('versioned_response'):
  d=json.loads((BASE/a['versioned_response']).read_text())
  if 'results' in d:
   baseline=json.loads((C/'raw/initial-openalex-citing.json').read_text());known={r['id'] for r in baseline['results']};academic.update(count=len(d['results']),new=[r for r in d['results'] if r['id'] not in known],pagination=d.get('meta',{}))
  elif 'data' in d:
   baseline=json.loads((C/'raw/initial-s2-citing.json').read_text());known={r['citingPaper']['paperId'] for r in baseline['data']};academic.update(count=len(d['data']),new=[r for r in d['data'] if r['citingPaper']['paperId'] not in known],next=d.get('next'))
  else:pending.append(dict(kind='academic',reason='response is not a citation-list shape'))
 else:pending.append(dict(kind='academic',reason='no usable independent academic response; preserve limitation, do not fabricate a zero count'))
 if academic['new']:pending.extend(dict(kind='academic-new-identity',record=x) for x in academic['new'])
 result=dict(round=n,reviewed_at=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),baseline_candidate_count=len(resources),web_queries=web.get('queries'),web_rows=web_rows,code_query=code.get('query'),code_rows=code_rows,academic=academic,pending=pending,new_eligible_resources=0 if not pending else None,zero_new_review_passed=not pending,gate1_passed=False,input_hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [web_file,code_file,C/'candidate-resources.json']})
 path=C/f'saturation-round{n}-review.json';path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print('round',n,'web',len(web_rows),'code',len(code_rows),'academic',academic.get('count'),'pending',len(pending),'zero-new passed',not pending)
 for r in pending:print(json.dumps(r,ensure_ascii=False)[:1100])
if __name__=='__main__':main()
