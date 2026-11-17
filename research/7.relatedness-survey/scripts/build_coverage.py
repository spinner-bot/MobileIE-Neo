"""Build a factual collection coverage matrix; this script does NOT approve a Gate."""
import datetime,json,hashlib,collections
from pathlib import Path
from zoneinfo import ZoneInfo
BASE=Path(__file__).resolve().parents[1];C=BASE/'collection'
def main():
 sources={
 'A-scholarly':[dict(name='Semantic Scholar DOI citation list',evidence=['receipts/initial-s2-citing.json','raw/initial-s2-citing.json','receipts/supp2-s2-list.json'],result='36 records; offset 0, no next page in successful response; arXiv alias request was rate limited'),dict(name='OpenAlex referenced_works citation list',evidence=['receipts/initial-openalex-citing.json','raw/initial-openalex-citing.json','receipts/oa-count-discrepancy-check.json'],result='40 list results, pagination exhausted; work metadata reports 41, discrepancy retained rather than assumed reconciled'),dict(name='Crossref',evidence=['receipts/initial-crossref.json','raw/initial-crossref.json'],result='Metadata count 16; not a forward-citation list'),dict(name='Google Scholar',evidence=['receipts/scholar-exact.json'],result='403; no claim of Scholar exhaustive coverage')],
 'B-official-and-code':[dict(name='Original publication/preprint/supplement and repository',evidence=['receipts/initial-cvf.json','receipts/official-paper-pdf.json','receipts/official-supplement-large.json','receipts/official-commits.json','receipts/official-issues.json'],result='Primary versions, full original PDF/supplement and official pinned source acquired'),dict(name='Repository-name, fork-tree and file-content discovery',evidence=['github-plugin-initial.json','code-duplicate-ledger.json','code-file-discovery-ledger.json','plugin-code-coverage-2.json','plugin-code-coverage-3.json'],result='Broader queries hit provider cap of 100 even with topn=300; exact-title query 75, MBRConv+MobileIE 71, MobileIELLENetS 18, temporal 39, flicker 17, TFLite 33; source limits disclosed, not claimed to enumerate all GitHub')],
 'C-authors-institutions-industry':[dict(name='Original author and group pages',evidence=['receipts/hailong-home.json','receipts/aoli-home.json','receipts/cezhu-publications.json','receipts/lezhang-publications.json','receipts/aoli-cqnu-home.json'],result='Available pages archived; Le Zhang website transport failure recorded'),dict(name='TeleAI',evidence=['receipts/teleai-primary-html.json','plugin-repo-Cloudyu1215--UIE.json'],result='Primary arXiv2603.16363 affiliation explicitly names TeleAI/China Telecom; method novelty/video applicability deferred to Gate2')],
 'D-chinese-english-web':[dict(name='Open web query matrix',evidence=[f.name for f in sorted(C.glob('web-initial-*.json'))]+['web-lead-review.json'],result='160 initial URL leads identity-reviewed; original papers/projects, aliases, directories and unrelated homonyms distinguished'),dict(name='Chinese academic entrances',evidence=['web-initial-3.json','receipts/cnki-exact.json','receipts/wanfang-exact.json'],result='CNKI transport failure; Wanfang returns client-rendered page, not verified complete query results; web site-qualified searches are an additional attempt, not proof of Chinese fulltext coverage')],
 'E-historical':[dict(name='Old S2 and OpenAlex lists',evidence=['historical-reconciliation.json','paper-candidates.json'],result='All 33 historical S2 and 18 historical OA records mapped into current candidate identities; old conclusions not adopted as new evidence')]
 }
 resources=json.loads((C/'candidate-resources.json').read_text());receipts={f.stem:json.loads(f.read_text()) for f in (C/'receipts').glob('*.json')}
 available=[]
 for r in resources:
  evidence=[]
  if r['temporary_key'].startswith('paper-'):
   evidence=[('receipts/'+key+'.json',x) for key,x in receipts.items() if key.startswith(r['temporary_key']+'-')]
  else:
   for src in r['sources']:
    name=src.get('acquisition') if isinstance(src,dict) else src
    if name and name.startswith('receipts/'):
     f=C/name
     assert f.exists(),name
     evidence.append((name,json.loads(f.read_text())))
   if r.get('repo')=='Barry1749/MobileIE':evidence.extend(('receipts/'+key+'.json',x) for key,x in receipts.items() if key.startswith('barry-text-'))
  available.append(dict(temporary_key=r['temporary_key'],type=r['type'],receipts=[dict(path=f,status=x['status'],http=x.get('http_status'),archive=x.get('archive_path'),sha256=x.get('sha256')) for f,x in evidence],adequacy='NOT reviewed: transport/file availability is not Gate2 analysis',supplementary_sources=r['sources']))
 freeze=json.loads((BASE/'gates/freeze-v1.json').read_text());gate_ok=all(hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256'] for x in freeze['files'])
 out=dict(recorded_at=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),source_matrix=sources,candidate_counts=dict(collections.Counter(r['type'] for r in resources)),gate_hashes_unchanged=gate_ok,gate1_status='NOT_APPROVED',known_limitations=['Snapshot only, not mathematical completeness of the web','Scholar403, CNKI failure, Wanfang client rendering and restricted publishers','GitHub broad search hard-capped; complementary exact queries reduce but cannot prove no omissions','OpenAlex metadata41/list40 mismatch; list40 reproduced through distinct API query; no imaginary 41st item added'])
 (C/'coverage-matrix.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');(C/'resource-acquisitions.json').write_text(json.dumps(available,ensure_ascii=False,indent=2)+'\n')
 print('Coverage entries',sum(len(v) for v in sources.values()),'candidate availability records',len(available),'Gate hashes',gate_ok)
if __name__=='__main__':main()
