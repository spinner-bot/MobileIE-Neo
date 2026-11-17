"""Reconcile all URL identities from recorded search responses without value grading."""
import json,re,unicodedata
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];C=BASE/'collection'
def norm(s):return re.sub(r'[^a-z0-9]','',unicodedata.normalize('NFKD',s).lower())
def main():
 initial={x['url']:x for x in json.loads((C/'web-lead-review.json').read_text())};resources=json.loads((C/'candidate-resources.json').read_text());papers=[r for r in resources if r['type']=='paper'];repos={r.get('repo'):r['temporary_key'] for r in resources if r.get('repo')};rows={}
 manual=json.loads((C/'supplement-web-identity-resolutions.json').read_text()) if (C/'supplement-web-identity-resolutions.json').exists() else {}
 for f in list(sorted(C.glob('web-initial-*.json')))+list(sorted(C.glob('web-supplement-round*.json')))+[C/'web-new-review-trace.json']:
  for b in json.loads(f.read_text())['result'].get('content',[]):
   if b.get('type')!='text':continue
   for e in b['text'].split('--------------------------------------------------------------------------------'):
    m=re.match(r'(.+) \((https?://[^\s]+)\)$',e.strip().split('\n')[0])
    if not m:continue
    title,url=m.groups()
    if url in rows:rows[url]['discoveries'].append(f.name);continue
    r=dict(title=title,url=url,discoveries=[f.name])
    if url in manual:r.update(manual[url])
    elif url in initial:r.update({k:initial[url][k] for k in ['disposition','reason']});r['resource']=initial[url].get('resource')
    else:
     candidates=[p for p in papers if len(norm(p['title']))>25 and (norm(p['title']) in norm(title) or norm(title) in norm(p['title']) and len(norm(title))>55)]
     if candidates:r.update(disposition='canonical-paper-alias-or-derivative-summary',resource=candidates[0]['temporary_key'],reason='Same paper identity/title; translated summaries, mirrored originals and bibliographic pages do not add an independent study')
     elif 'github.com/' in url and '/'.join(url.split('github.com/')[1].split('/')[:2]) in repos:r.update(disposition='known-project-alias',resource=repos['/'.join(url.split('github.com/')[1].split('/')[:2])],reason='Included project or its documented subresource')
     elif any(t.lower() in title.lower() for t in ['Hailong Yan','Ce Zhu','Zenglin Shi','Le Zhang','Ao Li','Zhe Liu','Xiangtao Zhang']):r.update(disposition='author-directory',reason='Original author identity/directory; no independent new technical artifact in this hit')
     elif 'ResearchSquare' in title or 'Sciety' in title:r.update(disposition='pending',reason='Needs identity resolution')
     else:r.update(disposition='pending',reason='Needs explicit identity/scope resolution')
    rows[url]=r
 (C/'all-web-lead-review.json').write_text(json.dumps(list(rows.values()),ensure_ascii=False,indent=2)+'\n')
 pending=[r for r in rows.values() if r['disposition']=='pending'];print('URL identities',len(rows),'pending',len(pending))
 for r in pending:print(r['title'],r['url'])
if __name__=='__main__':main()
