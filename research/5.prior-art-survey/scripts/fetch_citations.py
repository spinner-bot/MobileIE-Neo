"""抓取 MobileIE 的引用元数据（Crossref / OpenAlex / Semantic Scholar）。

用法：
    python research/5.prior-art-survey/scripts/fetch_citations.py \
        --doi 10.1109/ICCV51701.2025.02038 \
        --out research/5.prior-art-survey/data

输出：
    crossref_work.json          Crossref 元数据（含被引计数）
    openalex_work.json          OpenAlex 元数据（含引用计数与 OpenAlex ID）
    citations_openalex.json     OpenAlex 引用列表（含作者机构）
    citations_s2.json           Semantic Scholar 引用列表 + 摘要
    fetch_summary.json          本次抓取的计数与时间

注意：S2 / OpenAlex 对匿名请求有速率限制；脚本会自动退避重试。
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.request

UA = 'MobileIE-Neo-prior-art-survey/1.0 (research use)'


def get(url, tries=5, sleep=10):
    for i in range(tries):
        req = urllib.request.Request(url, headers={'User-Agent': UA})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and i < tries - 1:
                time.sleep(sleep * (i + 1))
                continue
            raise
    raise RuntimeError('request failed: %s' % url)


def post(url, payload, tries=5, sleep=10):
    body = json.dumps(payload).encode('utf-8')
    for i in range(tries):
        req = urllib.request.Request(url, data=body, headers={
            'User-Agent': UA, 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and i < tries - 1:
                time.sleep(sleep * (i + 1))
                continue
            raise
    raise RuntimeError('request failed: %s' % url)


def save(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print('saved', path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--doi', default='10.1109/ICCV51701.2025.02038')
    ap.add_argument('--out', default='research/5.prior-art-survey/data')
    ap.add_argument('--mailto', default='research@mobileie-neo.local')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    summary = {'doi': args.doi, 'retrieved_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}

    # 1) Crossref 元数据
    cr = get('https://api.crossref.org/works/%s' % args.doi)['message']
    save(cr, os.path.join(args.out, 'crossref_work.json'))
    summary['crossref_cited_by'] = cr.get('is-referenced-by-count')

    # 2) OpenAlex 元数据与引用列表
    oa = get('https://api.openalex.org/works/https://doi.org/%s?mailto=%s'
             % (args.doi, args.mailto))
    save(oa, os.path.join(args.out, 'openalex_work.json'))
    summary['openalex_id'] = oa['id'].rsplit('/', 1)[-1]
    summary['openalex_cited_by'] = oa.get('cited_by_count')
    cite_url = ('https://api.openalex.org/works?filter=cites:%s&per-page=50&mailto=%s'
                '&select=id,display_name,publication_year,doi,type,authorships,primary_location'
                % (summary['openalex_id'], args.mailto))
    oa_cites = get(cite_url)
    save(oa_cites, os.path.join(args.out, 'citations_openalex.json'))
    summary['openalex_citations_fetched'] = len(oa_cites.get('results', []))

    # 3) Semantic Scholar 引用列表 + 摘要
    s2_url = ('https://api.semanticscholar.org/graph/v1/paper/DOI:%s/citations'
              '?fields=title,year,venue,externalIds,authors,abstract&limit=100' % args.doi)
    s2 = get(s2_url)
    ids = [it['citingPaper']['paperId'] for it in s2.get('data', []) if it.get('citingPaper')]
    batch = post('https://api.semanticscholar.org/graph/v1/paper/batch'
                 '?fields=title,abstract,year,venue,authors,externalIds,publicationDate',
                 {'ids': ids}) if ids else []
    by_id = {p['paperId']: p for p in batch if p}
    papers = []
    for it in s2.get('data', []):
        c = it.get('citingPaper') or {}
        p = by_id.get(c.get('paperId'), {})
        for k in ('abstract', 'authors', 'externalIds', 'venue', 'year'):
            if p.get(k):
                c[k] = p[k]
        papers.append(c)
    save(papers, os.path.join(args.out, 'citations_s2.json'))
    save(s2, os.path.join(args.out, 'citations_s2_raw.json'))
    summary['s2_citations_fetched'] = len(papers)

    save(summary, os.path.join(args.out, 'fetch_summary.json'))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
