"""Official synthetic-clean StableLLVE weights, verified against pinned Git objects."""
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
    vendor=ROOT/'third_party/stablellve';upstream=json.loads((vendor/'UPSTREAM.json').read_text())
    item=next(r for r in upstream['files'] if r['path']=='checkpoint.pth')
    path=ROOT/'.cache/pretrained/stablellve'/upstream['commit']/item['path']
    url=f"https://raw.githubusercontent.com/zkawfanx/StableLLVE/{upstream['commit']}/{item['path']}"
    path.parent.mkdir(parents=True,exist_ok=True)
    payload=path.read_bytes() if path.exists() else read(url)
    blob=hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()
    if blob!=item['git_blob_sha1'] or len(payload)!=item['bytes']:raise ValueError('official checkpoint mismatch')
    if not path.exists():
        partial=path.with_suffix('.partial');partial.write_bytes(payload);partial.rename(path)
    sources=[]
    for name in ['LICENSE','model.py']:
        source=vendor/name;raw=source.read_bytes();expected=next(r for r in upstream['files'] if r['path']==name)
        digest=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        if digest!=expected['git_blob_sha1'] or len(raw)!=expected['bytes']:raise ValueError('source bytes changed')
        sources.append({'path':str(source.relative_to(ROOT)),'sha256':sha256(source),'git_blob_sha1':digest,'bytes':len(raw)})
    save_json(out/'manifest.json',{'repository':upstream['repository'],'commit':upstream['commit'],'url':url,
        'path':str(path.relative_to(ROOT)),'sha256':sha256(path),'git_blob_sha1':blob,'bytes':len(payload),'sources':sources,
        'upstream_manifest_sha256':sha256(vendor/'UPSTREAM.json'),
        'scope':'official synthetic-clean training, unsuitable-for-noise caveat retained; BGR source convention; no new training'})
    print(json.dumps({'bytes':len(payload),'sha256':sha256(path),'git_blob_sha1':blob}),flush=True)


if __name__=='__main__':main()
