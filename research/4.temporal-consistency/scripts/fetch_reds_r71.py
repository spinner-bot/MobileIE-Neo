"""Fetch only the committed R71 PNG members, with no model or new degradation."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
import time
import zlib

from PIL import Image
from reds_archive_r71 import RemoteZip, MAX_RANGE

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'research/4.temporal-consistency'
CATALOG = BASE / 'process/exploration-r71/catalog001'
OUT = BASE / 'process/exploration-r71/fetch001'
ASSETS = ROOT / '.cache/datasets/reds/r71/fetch001'


def stamp():
    return datetime.now().astimezone().isoformat()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    plan_path = CATALOG / 'selection.json'
    tracked = subprocess.check_output(['git', 'show', 'HEAD:' + str(plan_path.relative_to(ROOT))], cwd=ROOT)
    assert tracked == plan_path.read_bytes(), 'Selection must be committed and unchanged before payload'
    plan = json.loads(tracked)
    assert json.loads((CATALOG / 'runtime.json').read_text())['completed_normally']
    assert plan['members_total'] == len(plan['members']) == 256
    OUT.mkdir(parents=True, exist_ok=False)
    ASSETS.mkdir(parents=True, exist_ok=False)
    write(OUT / 'config.json', dict(started=stamp(), selection=str(plan_path.relative_to(ROOT)),
          selection_sha256=hashlib.sha256(tracked).hexdigest(), source=plan['source'],
          members=256, HTTP_workers=2, CNN=0, SGD=0, GPU=0,
          code_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [Path(__file__).resolve(), Path(__file__).with_name('reds_archive_r71.py')]}))
    started = time.perf_counter()
    runtime = dict(completed_normally=False, started=stamp(), completed_members=0,
                   CNN=0, SGD=0, GPU=0, nonselected_image_members_read=0,
                   whole_archive_downloaded=False, whole_archive_sha256_locally_verified=False)

    def fetch(row):
        remote = RemoteZip(plan['source']['archive_url'], plan['source']['archive_bytes'])
        # Read a single bounded span containing the local header and compressed member.
        extra_allowance = 4096
        wanted = 30 + len(row['name'].encode()) + extra_allowance + row['compressed_size']
        count = min(wanted, remote.size - row['local_offset'])
        assert 0 < count <= MAX_RANGE
        span = remote.range_bytes(row['local_offset'], count)
        fields = struct.unpack('<4s5H3I2H', span[:30])
        assert fields[0] == b'PK\x03\x04' and fields[3] == row['method']
        assert fields[2] == row['flags'] and not fields[2] & 1
        name_length, extra_length = fields[-2:]
        encoding = 'utf-8' if fields[2] & 2048 else 'cp437'
        assert span[30:30 + name_length].decode(encoding) == row['name']
        begin = 30 + name_length + extra_length
        end = begin + row['compressed_size']
        assert end <= len(span), 'Unexpected large local extra field; stop rather than change transfer bounds'
        compressed = span[begin:end]
        if row['method'] == 8:
            decoder = zlib.decompressobj(-15)
            value = decoder.decompress(compressed, row['size'] + 1)
            assert decoder.eof and not decoder.unconsumed_tail and not decoder.unused_data
        else:
            value = compressed
        assert len(value) == row['size'] and f'{zlib.crc32(value):08x}' == row['crc32']
        assert value.startswith(b'\x89PNG\r\n\x1a\n')
        with Image.open(io.BytesIO(value)) as image:
            assert image.mode == 'RGB' and list(image.size) == [1280, 720]
            image.load()  # Validate all decoded rows; do not convert, crop, rescale or expose a thumbnail.
        target = ASSETS / 'png' / Path(row['name']).parts[-2] / Path(row['name']).name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(value)
        return dict(member=row['name'], path=str(target.relative_to(ROOT)), bytes=len(value),
                    sha256=hashlib.sha256(value).hexdigest(), crc32=row['crc32'],
                    shape=[720, 1280, 3], dtype='uint8', mode='RGB', role='DEV',
                    range_bytes=remote.transferred, requests=remote.ranges, retries=remote.retries)

    try:
        # Submit at most two requests at once; stop submission immediately on a failed member.
        members = iter(plan['members'])
        with (OUT / 'manifest.jsonl').open('x') as manifest, ThreadPoolExecutor(max_workers=2) as pool:
            pending = [pool.submit(fetch, next(members)) for _ in range(2)]
            while pending:
                future = pending.pop(0)
                receipt = future.result()
                manifest.write(json.dumps(receipt) + '\n'); manifest.flush()
                runtime['completed_members'] += 1
                print(json.dumps(dict(at=stamp(), completed=runtime['completed_members'], member=receipt['member'])), flush=True)
                row = next(members, None)
                if row is not None:
                    pending.append(pool.submit(fetch, row))
        assert runtime['completed_members'] == 256
        runtime['completed_normally'] = True
    except Exception as error:
        runtime.update(error_type=type(error).__name__, error_status=getattr(error, 'code', None))
        raise
    finally:
        runtime.update(ended=stamp(), seconds=time.perf_counter() - started)
        write(OUT / 'runtime.json', runtime)
    print(json.dumps(runtime))


if __name__ == '__main__':
    main()
