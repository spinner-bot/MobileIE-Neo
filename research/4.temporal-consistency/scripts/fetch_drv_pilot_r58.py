"""Fetch exactly the frozen 36 public RAW ZIP members; never decode image pixels."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import http.client
import json
import struct
import time
import traceback
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zlib

from dynamic_training_r34 import BASE, ROOT, digest


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def read_range(meta, start, end, retries):
    expected = end-start+1
    assert 0 <= start <= end < meta['archive_bytes'] and expected <= 2*1024*1024
    for attempt in range(1, 4):
        try:
            request = Request(meta['url'], headers={'Range': f'bytes={start}-{end}',
                              'If-Match': meta['etag'], 'Accept-Encoding': 'identity'})
            with urlopen(request, timeout=60) as response:
                assert response.status == 206
                assert response.headers['Content-Range'] == f'bytes {start}-{end}/{meta["archive_bytes"]}'
                assert response.headers.get('ETag') == meta['etag']
                assert int(response.headers['Content-Length']) == expected
                value = response.read(expected+1)
            assert len(value) == expected
            return value
        except (URLError, TimeoutError, ConnectionError, http.client.IncompleteRead) as exc:
            if isinstance(exc, HTTPError) and exc.code < 500: raise
            retries.append(dict(url=meta['url'], start=start, end=end, attempt=attempt,
                                error=type(exc).__name__, message=str(exc)[:300]))
            if attempt == 3: raise
            time.sleep(attempt)


def fetch_one(row, meta, cache, retries):
    assert row['name'].lower().endswith('.arw') and 0 < row['size'] < 25*1024*1024
    assert 0 < row['compressed_size'] < 25*1024*1024 and row['method'] in [0, 8]
    target = cache/row['output']; target.parent.mkdir(parents=True, exist_ok=True)
    assert target.resolve().is_relative_to(cache.resolve())
    header = read_range(meta, row['offset'], row['offset']+29, retries)
    fields = struct.unpack('<4s5H3I2H', header)
    assert fields[0] == b'PK\x03\x04' and fields[3] == row['method'] and not fields[2] & 1
    namesize, extrasize = fields[-2:]
    assert 0 < namesize <= 512 and namesize+extrasize <= 8192
    name = read_range(meta, row['offset']+30, row['offset']+29+namesize, retries)
    assert name.decode('utf-8' if fields[2] & 2048 else 'cp437') == row['name']
    start = row['offset']+30+namesize+extrasize
    compressed = target.with_suffix('.ARW.zipdata')
    partial = compressed.with_suffix('.zipdata.part')
    with partial.open('xb') as stream:
        for offset in range(0, row['compressed_size'], 2*1024*1024):
            size = min(2*1024*1024, row['compressed_size']-offset)
            stream.write(read_range(meta, start+offset, start+offset+size-1, retries))
    assert partial.stat().st_size == row['compressed_size']; partial.rename(compressed)
    raw = target.with_suffix('.ARW.part')
    decoder = zlib.decompressobj(-zlib.MAX_WBITS) if row['method'] == 8 else None
    crc, size, sha = 0, 0, hashlib.sha256()
    with compressed.open('rb') as source, raw.open('xb') as output:
        def emit(value):
            nonlocal crc, size
            size += len(value); assert size <= row['size']
            crc = zlib.crc32(value, crc); sha.update(value); output.write(value)
        for block in iter(lambda: source.read(1024*1024), b''):
            emit(decoder.decompress(block) if decoder else block)
        if decoder:
            emit(decoder.flush()); assert decoder.eof and not decoder.unused_data
    assert size == row['size'] and f'{crc:08x}' == row['crc32']; raw.rename(target)
    return dict(scene=row['scene'], role=row['role'], ordinal=row['ordinal'], archive=row['archive'],
                member=row['name'], crc32=row['crc32'], sha256=sha.hexdigest(), bytes=size,
                path=str(target.relative_to(ROOT)), compressed_path=str(compressed.relative_to(ROOT)),
                compressed_bytes=row['compressed_size'], decoded_pixels=False)


def main():
    plan_path = BASE/'data/external-drv/raw-pilot-plan-r58.json'; plan = json.loads(plan_path.read_text())
    assert len(plan['members']) == 36
    assert digest(BASE/'docs/exploration-r58-drv-static-raw-pilot-protocol.md') == plan['protocol_sha256']
    out = BASE/'process/external-drv/raw-acquisition001'; out.mkdir(exist_ok=False)
    cache = ROOT/'.cache/datasets/DRV/raw-pilot/attempt001'
    assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    cache.mkdir(parents=True, exist_ok=False)
    write(out/'config.json', dict(at=datetime.now().astimezone().isoformat(), plan_sha256=digest(plan_path),
          source_sha256=digest(BASE/'scripts/fetch_drv_pilot_r58.py'), workers=4,
          range_chunk_bytes=2*1024*1024, technical_retries_per_range=3, decoded_pixels=0,
          selection='exact frozen members, no PNG/val/test/dynamic; no automatic decoding or CNN'))
    started, complete, results, retries = time.perf_counter(), False, [], []
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(fetch_one, row, plan['archives'][row['archive']], cache, retries)
                       for row in plan['members']]
            try:
                for future in as_completed(futures):
                    result = future.result(); results.append(result)
                    with (out/'members.jsonl').open('a') as stream:
                        stream.write(json.dumps(result, allow_nan=False)+'\n')
                    print(json.dumps(dict(completed=len(results), scene=result['scene'], member=result['member'])), flush=True)
            except BaseException:
                for future in futures: future.cancel()
                raise
        assert len(results) == 36 and sum(r['bytes'] for r in results) == plan['raw_bytes']
        write(out/'manifest.json', dict(members=sorted(results, key=lambda r: (r['scene'], r['role'], r['ordinal'])),
              total_raw_bytes=sum(r['bytes'] for r in results), total_compressed_bytes=plan['compressed_payload_bytes'],
              archives=plan['archives'], decoded_pixels=0, CNN=0, GPU=False))
        complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'technical-retries.json', retries)
        write(out/'runtime.json', dict(completed_normally=complete, complete_members=len(results),
              decoded_pixels=0, CNN=0, GPU=False, seconds=time.perf_counter()-started,
              ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__': main()
