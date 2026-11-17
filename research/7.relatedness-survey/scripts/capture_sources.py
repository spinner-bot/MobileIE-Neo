"""Read-only URL acquisition; preserve successes AND failures without executing sources.
Use --jobs JSON [{label,url,category}] and optional --refresh for a new dated capture.
All archives are zrh-owned, and every response has a hash and actual timestamp.
"""
import argparse, concurrent.futures, datetime, hashlib, json, os, re, urllib.request, urllib.error
from pathlib import Path
from zoneinfo import ZoneInfo
ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'research/7.relatedness-survey'
ARCHIVE = ROOT/'.cache/relatedness-survey/archive'
def capture(job):
    at=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    key=re.sub(r'[^a-zA-Z0-9_.-]', '_', job['label'])
    record=dict(job, requested_at=at, author='Codex  (sqs_A100)', request_method='GET')
    req=urllib.request.Request(job['url'],headers={'User-Agent':'MobileIE-Neo-relatedness-survey/1.0 (public research; langxibielangle@qq.com)'})
    data=b''
    try:
        with urllib.request.urlopen(req,timeout=25) as r:
            limit=job.get('max_bytes',24*1024*1024)
            data=r.read(limit+1)
            record.update(http_status=r.status,final_url=r.url,content_type=r.headers.get('Content-Type'),headers={k:v for k,v in r.headers.items() if k.lower() in ('etag','last-modified','content-length','content-type')})
            if len(data)>limit:
                record.update(status='size_limit_incomplete',complete=False); data=data[:limit]
            else:record.update(status='success',complete=True)
    except urllib.error.HTTPError as e:
        data=e.read(1024*1024)
        record.update(status='http_error',http_status=e.code,error=str(e),complete=False)
    except Exception as e:record.update(status='transport_error',error=repr(e),complete=False)
    if data:
        suffix='.pdf' if data.startswith(b'%PDF') else '.json' if data.lstrip()[:1] in (b'{',b'[') else '.html' if b'<html' in data[:1000].lower() or b'<!doctype' in data[:1000].lower() else '.txt'
        archive=ARCHIVE/(key+suffix); archive.write_bytes(data)
        record.update(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),archive_path=str(Path('archive')/archive.name))
        if suffix=='.json' and len(data)<3*1024*1024:
            (BASE/'collection/raw'/archive.name).write_bytes(data)
            record['versioned_response']='collection/raw/'+archive.name
    record['finished_at']=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    (BASE/'collection/receipts'/(key+'.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    return {'label':key,'status':record['status'],'http_status':record.get('http_status'),'bytes':record.get('bytes')}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--jobs',required=True);args=ap.parse_args()
    jobs=json.loads(Path(args.jobs).read_text())
    for folder in [ARCHIVE,BASE/'collection/raw',BASE/'collection/receipts']:folder.mkdir(parents=True,exist_ok=True)
    link=BASE/'archive'
    if not link.is_symlink():link.symlink_to('../../.cache/relatedness-survey/archive',target_is_directory=True)
    labels=[j['label'] for j in jobs]
    assert len(labels)==len(set(labels)), 'duplicate labels'
    for label in labels:
        assert not (BASE/'collection/receipts'/(label+'.json')).exists(), 'preserve old receipts: choose a new label'
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(capture,jobs):print(json.dumps(result),flush=True)
if __name__=='__main__':main()
