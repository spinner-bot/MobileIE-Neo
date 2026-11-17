"""Union current+historical citation records without grading or permanent IDs."""
import hashlib,json,re,unicodedata
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];ROOT=BASE.parents[1]
def titlekey(s):return re.sub(r'[^a-z0-9]','',unicodedata.normalize('NFKD',s).lower())
def doi(v):return (v or '').lower().removeprefix('https://doi.org/').removeprefix('http://doi.org/')
def main():
 rows=[]
 def ingest(kind,items,source):
  for i,it in enumerate(items):
   p=it.get('citingPaper',it)
   ext=p.get('externalIds') or {}
   if kind=='oa':
    d=doi(p.get('doi'));ar=re.search(r'arxiv\.(\d{4}\.\d{4,5})',d,re.I);ar=ar.group(1) if ar else None
    title=p.get('display_name','');authors=[a.get('author',{}).get('display_name') for a in p.get('authorships',[])];year=p.get('publication_year');url=p.get('primary_location',{}).get('landing_page_url');pdf=p.get('open_access',{}).get('oa_url');abstract=p.get('abstract_inverted_index');rid=p.get('id')
   else:
    d=doi(ext.get('DOI'));ar=ext.get('ArXiv');title=p.get('title','');authors=[a.get('name') for a in p.get('authors',[])];year=p.get('year');url=p.get('url');pdf=(p.get('openAccessPdf') or {}).get('url');abstract=p.get('abstract');rid=p.get('paperId')
   if title: rows.append(dict(title=title,doi=d,arxiv=ar,year=year,authors=authors,url=url,pdf=pdf,abstract=abstract,source=source,position=i,source_record_id=rid))
 for name,kind in [('initial-openalex-citing','oa'),('initial-s2-citing','s2')]:
  x=json.loads((BASE/'collection/raw'/f'{name}.json').read_text());ingest(kind,x['results'] if kind=='oa' else x['data'],f'collection/raw/{name}.json')
 for name,kind in [('citations_s2','s2'),('citations_openalex','oa')]:
  f=ROOT/'research/5.prior-art-survey/data'/f'{name}.json';x=json.loads(f.read_text());ingest(kind,x['results'] if isinstance(x,dict) else x,str(f.relative_to(ROOT)))
 groups=[]
 for row in rows:
  keys={titlekey(row['title'])}
  existing=[g for g in groups if titlekey(row['title']) in g['title_keys'] or row['doi'] and row['doi'] in g['dois'] or row['arxiv'] and row['arxiv'] in g['arxivs']]
  if len(existing)>1:raise ValueError('conflicting identity groups: '+row['title'])
  if existing:g=existing[0]
  else:
   g=dict(temporary_key='paper-'+hashlib.sha256(titlekey(row['title']).encode()).hexdigest()[:12],type='paper',title=row['title'],title_keys=[],dois=[],arxivs=[],years=[],authors=[],urls=[],discovery=[],relation='recorded as citing MobileIE by a current or historical citation index; exact citation context pending Gate2',status='collection_candidate');groups.append(g)
  for field,val in [('title_keys',titlekey(row['title'])),('dois',row['doi']),('arxivs',row['arxiv']),('years',row['year'])]:
   if val and val not in g[field]:g[field].append(val)
  for u in [row['url'],row['pdf']]:
   if u and u not in g['urls']:g['urls'].append(u)
  if row['authors'] and not g['authors']:g['authors']=row['authors']
  g['discovery'].append({k:row[k] for k in ['source','position','source_record_id']})
  if row['abstract']:g.setdefault('abstract_evidence',[]).append(dict(source=row['source'],value=row['abstract']))
 (BASE/'collection/paper-candidates.json').write_text(json.dumps(groups,ensure_ascii=False,indent=2)+'\n')
 jobs=[]
 for g in groups:
  key=g['temporary_key']
  urls=[]
  if g['arxivs']:
   ar=g['arxivs'][0];urls.extend([('arxiv-abs','https://arxiv.org/abs/'+ar),('arxiv-pdf','https://arxiv.org/pdf/'+ar),('arxiv-html','https://arxiv.org/html/'+ar)])
  if g['dois']:
   d=next((d for d in g['dois'] if not d.startswith('10.48550/')),g['dois'][0]);urls.append(('publisher','https://doi.org/'+d))
  for i,u in enumerate(g['urls']):
   if not any(u==v for _,v in urls):urls.append((f'index-link{i}',u))
  for suffix,url in urls:jobs.append(dict(label=key+'-'+suffix,url=url,category='candidate-acquisition',candidate=key))
 (BASE/'collection/paper-acquisition-jobs.json').write_text(json.dumps(jobs,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(dict(candidate_papers=len(groups),raw_records=len(rows),acquisition_jobs=len(jobs))))
 for g in groups:print(g['temporary_key'],g['title'])
if __name__=='__main__':main()
