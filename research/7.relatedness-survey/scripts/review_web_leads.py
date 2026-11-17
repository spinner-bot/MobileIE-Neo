"""Persist reviewed source identities, aliases and explicit exclusions from initial web hits.
Indices refer to the immutable initial discovery ledger; no value grades are assigned.
"""
import json
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];C=BASE/'collection'
def main():
 rows=json.loads((C/'web-discovery-ledger.json').read_text())
 anchors={0,1,3,4,6,7,10,11,12,14,16,17,18,19,20,22,24,25,30,33,35,36,38,39,51,52,53,61,127,155}
 directories={5,8,15,29,32,40,42,43,44,45,46,48,62,73,81,84,88,108,109,111,116,117,119,128,129,131,133,137,142,145,146,148,149,150,151,157}
 backwards={13,47,54,55,56,57,58,59,60,132,153}
 false_hits={49,65,71,75,89,90,91,92,93}
 reports={37:'related-rgb2025-report',50:'related-rgb2025-report',41:'related-ntire2025-report',94:'related-ntire2025-report',97:'related-ntire2025-report',105:'related-ntire2025-report',120:'related-ntire2025-report',134:'related-ntire2025-report',135:'related-ntire2025-report',136:'related-ntire2025-report',138:'related-ntire2025-report',139:'related-ntire2025-report',140:'related-ntire2025-report',141:'related-ntire2025-report',143:'related-ntire2025-report',144:'related-ntire2025-report',147:'related-ntire2025-report'}
 special={2:'code-AVC2-UESTC--MobileIE',9:'code-w-xb--MobileIE-6Ch',23:'code-w-xb--MobileIE-6Ch',26:'other-mobileie-explanations',27:'other-mobileie-explanations',28:'other-mobileie-explanations',31:'other-mobileie-explanations',34:'other-mobileie-explanations',63:'other-mobileie-rtdetr-port',64:'other-mobileie-rtdetr-port',66:'other-mobileie-rtdetr-port',67:'other-mobileie-explanations',68:'other-mobileie-reading-notes',69:'other-mobileie-reading-notes',72:'other-mobileie-rtdetr-port',74:'other-mobileie-explanations',76:'paper-f2b2063bbd33',77:'paper-dce553dc3465',78:'paper-dce553dc3465',79:'paper-dce553dc3465',80:'paper-dce553dc3465',82:'paper-dce553dc3465',83:'paper-dce553dc3465',85:'paper-dce553dc3465',86:'paper-dce553dc3465',87:'paper-dce553dc3465',95:'paper-abe88a6f55ac',96:'paper-abe88a6f55ac',98:'paper-57ace90aa47c',99:'code-lhfgghc--LLVE-AMNet',100:'paper-57ace90aa47c',101:'paper-57ace90aa47c',102:'paper-1bcaf3288599',103:'paper-57ace90aa47c',104:'paper-57ace90aa47c',106:'paper-57ace90aa47c',107:'paper-57ace90aa47c',110:'paper-57ace90aa47c',112:'paper-57ace90aa47c',113:'paper-57ace90aa47c',114:'paper-57ace90aa47c',115:'paper-1bcaf3288599',118:'paper-1bcaf3288599',121:'paper-1bcaf3288599',123:'paper-1bcaf3288599',124:'paper-1bcaf3288599',130:'paper-abe88a6f55ac',154:'paper-1bcaf3288599',156:'paper-1bcaf3288599',158:'paper-c98e473fe94c',159:'paper-1bcaf3288599'}
 lists={122,125,126,152}
 for i,x in enumerate(rows):
  x['initial_ledger_position']=i
  if i in anchors:x.update(disposition='canonical-alias-or-bibliographic-evidence',resource='anchor-mobileie-paper',reason='Same original MobileIE publication or official artifact; preserve discovery, no independent-paper count')
  elif i==21:x.update(disposition='excluded-administrative',reason='GitHub security overview contains no research resource to analyze; no security operations performed')
  elif i in directories:x.update(disposition='discovery-directory',reason='Author/institution/topic/search directory, used for tracing linked works rather than counted as an independent technical contribution')
  elif i in backwards:x.update(disposition='excluded-backward-bibliography-only',reason='Search matches a paper cited by MobileIE or a bibliographic co-occurrence, not a new direct citing/derivative project; retain as contextual lead, not claim unrelated scientifically')
  elif i in false_hits:x.update(disposition='excluded-no-demonstrated-direct-link',reason='Generic similarity recommendation, unrelated mobile software/patent, or page without MobileIE identity; no evidence of a direct citing or derived artifact within this survey scope')
  elif i in reports:x.update(disposition='associated-report-or-alias',resource=reports[i],reason='Official or mirrored report identity traced to included original; derivative host not independent confirmation')
  elif i in special:x.update(disposition='included-or-canonical-alias',resource=special[i],reason='Identity matched included candidate; fulltext/code content and value are deferred until after Gate1')
  elif i in lists:x.update(disposition='bibliographic-discovery-list',resource='paper-1bcaf3288599',reason='Reading-list/daily-index entry for an already included paper; no new study beyond the listed source identity')
  elif i==70:x.update(disposition='paper-summary-alias',resource='paper-57ace90aa47c',reason='Secondary explanation of included AnyMod paper; original PMLR/arXiv available')
  else:raise AssertionError(f'unreviewed lead {i}: {x["title"]}')
  x['review_scope']='Collection identity/inclusion only; not Gate2 deep analysis'
 (C/'web-lead-review.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
 print('Reviewed',len(rows),'initial web leads; all dispositions explicit')
if __name__=='__main__':main()
