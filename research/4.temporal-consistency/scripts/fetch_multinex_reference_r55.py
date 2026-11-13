"""Acquire the immutable research-only Nano sources and one official checkpoint."""
import argparse
import hashlib
import json
import time
import traceback
import urllib.request
from datetime import datetime

from dynamic_training_r34 import BASE, ROOT, digest


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--attempt', type=int, required=True)
    args = parser.parse_args(); assert args.attempt >= 1
    plan_path = BASE/'data/external-multinex/acquisition-plan.json'
    plan = json.loads(plan_path.read_text())
    out = BASE/f'process/external-multinex/acquisition{args.attempt:03d}'
    out.mkdir(exist_ok=False)
    vendor = ROOT/'third_party/multinex'
    weights = ROOT/'.cache/weights/multinex/LOL-v1'
    assert weights.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    start, complete, rows = time.perf_counter(), False, []
    try:
        for item in plan['files']:
            url = f"https://raw.githubusercontent.com/{plan['repository']}/{plan['commit']}/{item['path']}"
            path = weights/'Multinex-Nano_LOL-v1.pth' if item['role'] == 'weight' else vendor/item['path']
            reused = path.exists()
            if reused:
                value = path.read_bytes()
            else:
                with urllib.request.urlopen(url, timeout=30) as response:
                    assert response.status == 200
                    value = response.read(item['bytes']+1)
            assert len(value) == item['bytes']
            git_blob = hashlib.sha1(f'blob {len(value)}\0'.encode()+value).hexdigest()
            assert git_blob == item['git_blob_sha'], item['path']
            if not reused:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as stream: stream.write(value)
            row = dict(source=item['path'], role=item['role'], url=url, path=str(path.relative_to(ROOT)),
                       bytes=len(value), git_blob_sha=git_blob, sha256=hashlib.sha256(value).hexdigest(),
                       reused_verified_file=reused)
            rows.append(row)
            print(json.dumps(dict(path=row['path'], bytes=len(value), verified=True)), flush=True)
        assert len(rows) == 11 and sum(r['role'] == 'weight' for r in rows) == 1
        metadata = dict(repository=plan['repository'], commit=plan['commit'], files=rows,
              acquisition_plan=str(plan_path.relative_to(ROOT)), plan_sha256=digest(plan_path),
              license='Original PolyForm Noncommercial1.0.0 and notices retained; non-commercial research only.',
              modifications='None; official source/config/test/loader retained byte for byte.',
              weight_execution=False, new_CNN=0, new_GPU=False)
        write(out/'manifest.json', metadata)
        write(vendor/'UPSTREAM.json', metadata)
        complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(completed_normally=complete, files_verified=len(rows),
              seconds=time.perf_counter()-start, ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__':
    main()
