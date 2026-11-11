"""Fetch author-hosted PDF in validated byte ranges after a recorded truncated read."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request

URL='https://www.cse.cuhk.edu.hk/~byu/papers/C131-ICCV2021-SDSD.pdf'
TOTAL=16970584
ETAG='"102f358-5e7ab31736a46"'


def main():
    cache=Path('.cache/source-papers/SDSD-parts');cache.mkdir(parents=True,exist_ok=True)
    def fetch(start):
        end=min(TOTAL-1,start+262144-1);path=cache/f'{start:09d}.bin'
        if path.exists():
            data=path.read_bytes()
            if len(data)!=end-start+1:raise ValueError('incomplete cached range')
        else:
            for attempt in range(3):
                try:
                    request=urllib.request.Request(URL,headers={'Range':f'bytes={start}-{end}','If-Range':ETAG})
                    with urllib.request.urlopen(request,timeout=60) as response:
                        if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{TOTAL}' or response.headers.get('ETag')!=ETAG:
                            raise ValueError('PDF range or source version changed')
                        data=response.read()
                    if len(data)!=end-start+1:raise ValueError('PDF range truncated')
                    path.write_bytes(data);break
                except Exception as error:
                    print(json.dumps({'start':start,'attempt':attempt+1,'error':str(error)}),flush=True)
                    if attempt==2:raise
                    time.sleep(attempt+1)
        return {'start':start,'end':end,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'path':str(path)}
    with ThreadPoolExecutor(max_workers=4) as executor:parts=list(executor.map(fetch,range(0,TOTAL,262144)))
    payload=b''.join(Path(r['path']).read_bytes() for r in parts)
    if len(payload)!=TOTAL or not payload.startswith(b'%PDF'):raise ValueError('assembled PDF invalid')
    path=Path('.cache/source-papers/SDSD-ICCV2021.pdf');path.write_bytes(payload)
    out=Path('research/5.prior-art-survey/process/data-readiness-r14/SDSD-paper-source.json')
    out.write_text(json.dumps({'url':URL,'etag':ETAG,'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest(),
        'path':str(path),'parts':parts,'scope':'author-hosted PDF; exact Content-Range/ETag checked; original truncated transfer recorded separately'},indent=2)+'\n')
    print(json.dumps({'parts':len(parts),'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}),flush=True)


if __name__=='__main__':main()
