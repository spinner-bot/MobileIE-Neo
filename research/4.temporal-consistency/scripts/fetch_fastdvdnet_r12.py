"""Download the pinned official binary, verifying Git object and local SHA256."""
import argparse
import hashlib
import json
from pathlib import Path

from exploration_r2_data import ROOT
from explore_temporal_controls import save_json, sha256
from fetch_olives_r4 import read


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    out=Path(p.parse_args().out);out.mkdir(parents=True,exist_ok=False)
    vendor=ROOT/'third_party/fastdvdnet';upstream=json.loads((vendor/'UPSTREAM.json').read_text())
    item=next(r for r in upstream['files'] if r['path']=='model_clipped_noise.pth')
    path=ROOT/'.cache/pretrained/fastdvdnet'/upstream['commit']/item['path']
    url=f"https://raw.githubusercontent.com/m-tassano/fastdvdnet/{upstream['commit']}/{item['path']}"
    path.parent.mkdir(parents=True,exist_ok=True)
    payload=path.read_bytes() if path.exists() else read(url)
    digest=hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()
    if len(payload)!=item['bytes'] or digest!=item['git_blob_sha1']:raise ValueError('official weight integrity mismatch')
    if not path.exists():
        partial=path.with_suffix('.partial');partial.write_bytes(payload);partial.rename(path)
    sources=[]
    for name in ['LICENSE','models.py','fastdvdnet.py']:
        source=vendor/name;raw=source.read_bytes();r=next(r for r in upstream['files'] if r['path']==name)
        blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        if blob!=r['git_blob_sha1'] or len(raw)!=r['bytes']:raise ValueError('vendored source changed')
        sources.append({'path':str(source.relative_to(ROOT)),'sha256':sha256(source),'git_blob_sha1':blob,'bytes':len(raw)})
    save_json(out/'manifest.json',{'repository':upstream['repository'],'commit':upstream['commit'],'url':url,
        'path':str(path.relative_to(ROOT)),'sha256':sha256(path),'git_blob_sha1':digest,'bytes':len(payload),
        'sources':sources,'upstream_manifest_sha256':sha256(vendor/'UPSTREAM.json'),
        'scope':'official clipped-AWGN weights; no task adaptation; binary fetched over HTTPS because connector rejects binary'})
    print(json.dumps({'bytes':len(payload),'sha256':sha256(path),'git_blob_sha1':digest}),flush=True)


if __name__=='__main__':main()
