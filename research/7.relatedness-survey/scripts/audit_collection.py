"""Check actual archive integrity and distinguish HTTP transport from usable material."""
import collections,hashlib,json,re
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
def main():
 rows=[]
 for f in sorted((BASE/'collection/receipts').glob('*.json')):
  x=json.loads(f.read_text());r=dict(label=x['label'],receipt=str(f.relative_to(BASE)),transport=x['status'],http=x.get('http_status'),usable=False,reason='no archived response')
  if x.get('archive_path'):
   p=BASE/x['archive_path'];data=p.read_bytes();assert hashlib.sha256(data).hexdigest()==x['sha256'],str(p)
   assert len(data)==x['bytes']
   if x['status']!='success':r['reason']=x['status']
   elif x.get('http_status')!=200:r['reason']='non-200 response, not assumed to be material'
   elif not data:r['reason']='empty response'
   elif re.search(br'(?i)<title[^>]*>\s*(just a moment|access denied|security|robot|captcha|challenge)',data[:5000]):r['reason']='HTML access challenge'
   else:r.update(usable=True,reason='nonempty HTTP200 acquisition; content-specific adequacy still requires reading',file=x['archive_path'],bytes=len(data),format='pdf' if data.startswith(b'%PDF') else 'json' if data.lstrip()[:1] in (b'[',b'{') else 'text/html')
  rows.append(r)
 out=dict(archive_checks=len(rows),counts=dict(collections.Counter(x['transport'] for x in rows)),usable_acquisitions=sum(x['usable'] for x in rows),records=rows,scope='Integrity and transport audit ONLY; not Gate1 approval or fulltext adequacy proof')
 (BASE/'collection/acquisition-audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:v for k,v in out.items() if k!='records'}))
 for r in rows:
  if not r['usable']:print(r['label'],r['http'],r['reason'])
if __name__=='__main__':main()
