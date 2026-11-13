"""Fetch only the frozen128 original PNGs, with ZIP CRC, no image/CNN decoding."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import struct
import sys
import time
import traceback
import zlib

from sdsd_archive_r67 import BASE, ROOT, MAX_RANGE, SplitArchive, digest, write

sys.path.insert(0, str(ROOT/'scripts'))
from project_assets import mounted_data


def fetch_one(row, volumes, cache):
    remote = SplitArchive(volumes); disk, offset = row['disk'], row['local_offset']
    header = remote.read_span(disk, offset, 30); fields = struct.unpack('<4s5H3I2H', header)
    assert fields[0] == b'PK\x03\x04' and fields[3] == row['method'] and not fields[2] & 1
    namesize, extrasize = fields[-2:]; assert 0 < namesize <= 512 and namesize+extrasize <= 8192
    name = remote.read_span(disk, offset+30, namesize)
    assert name.decode('utf-8' if fields[2] & 2048 else 'cp437') == row['name']
    start = offset+30+namesize+extrasize
    path = cache/row['output']; path.parent.mkdir(parents=True, exist_ok=True)
    assert path.resolve().is_relative_to(cache.resolve())
    compressed = path.with_suffix('.png.zipdata'); partial = compressed.with_suffix('.zipdata.part')
    with partial.open('xb') as stream:
        for position in range(0, row['compressed_size'], MAX_RANGE):
            stream.write(remote.read_span(disk, start+position, min(MAX_RANGE, row['compressed_size']-position)))
    assert partial.stat().st_size == row['compressed_size']; partial.rename(compressed)
    raw = path.with_suffix('.png.part'); decoder = zlib.decompressobj(-zlib.MAX_WBITS) if row['method'] == 8 else None
    crc, size, sha = 0, 0, hashlib.sha256()
    with compressed.open('rb') as source, raw.open('xb') as target:
        def emit(value):
            nonlocal crc, size
            size += len(value); assert size <= row['size']
            crc = zlib.crc32(value, crc); sha.update(value); target.write(value)
        for block in iter(lambda: source.read(MAX_RANGE), b''): emit(decoder.decompress(block) if decoder else block)
        if decoder:
            emit(decoder.flush()); assert decoder.eof and not decoder.unused_data
    assert size == row['size'] and f'{crc:08x}' == row['crc32']; raw.rename(path)
    result = dict(domain=row['domain'], scene=row['scene'], pair=row['pair'], role=row['role'], ordinal=row['ordinal'],
        member=row['name'], path=str(path.relative_to(ROOT)), sha256=sha.hexdigest(), bytes=size, crc32=row['crc32'],
        source_disk=disk, source_local_offset=offset, compressed_path=str(compressed.relative_to(ROOT)),
        compressed_bytes=row['compressed_size'], transferred_bytes=remote.transferred,
        technical_retries=remote.retries, observed_etags=remote.observed_etags, image_decoded=False)
    receipt = cache/'receipts'/f'{row["domain"]}-{row["pair"]}-{row["role"]}-{row["ordinal"]:02d}.json'
    receipt.parent.mkdir(exist_ok=True); write(receipt, result)
    return result


def main():
    mounted_data()
    plan_path = BASE/'data/exploration-r67-sdsd/selection.json'; plan = json.loads(plan_path.read_text())
    assert len(plan['members']) == 128 and len(plan['pairing_table']) == 4
    assert digest(BASE/'docs/exploration-r67-sdsd-fixed-acquisition-protocol.md') == plan['protocol_sha256']
    out = BASE/'process/exploration-r67/acquisition001'; out.mkdir(exist_ok=False)
    cache = ROOT/'.cache/datasets/SDSD/pilot-r67/attempt001'
    assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    cache.mkdir(parents=True, exist_ok=False)
    write(out/'config.json', dict(at=datetime.now().astimezone().isoformat(), selection_sha256=digest(plan_path),
        source_sha256={name: digest(BASE/'scripts'/name) for name in ['fetch_sdsd_r67.py', 'sdsd_archive_r67.py']},
        workers=4, max_range=MAX_RANGE, max_technical_attempts_per_range=3, member_count=128,
        pairing='Original independently sorted per-side arrays at equal ordinal, no equal-basename intersection.',
        image_decodes=0, enhancement_CNN=0, GPU=False))
    start, complete, results = time.perf_counter(), False, []
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            tasks = [pool.submit(fetch_one, row, plan['archives'][row['domain']]['volumes'], cache) for row in plan['members']]
            try:
                for task in as_completed(tasks):
                    result = task.result(); results.append(result)
                    with (out/'members.jsonl').open('a') as stream: stream.write(json.dumps(result)+'\n')
                    print(json.dumps(dict(complete_members=len(results), scene=result['scene'], role=result['role'], ordinal=result['ordinal'])), flush=True)
            except BaseException:
                for task in tasks: task.cancel()
                raise
        assert len(results) == 128 and sum(r['bytes'] for r in results) == plan['PNG_bytes']
        write(out/'manifest.json', dict(completed=True, source_selection_sha256=digest(plan_path),
            members=sorted(results, key=lambda r: (r['domain'], r['pair'], r['role'], r['ordinal'])),
            member_count=128, PNG_bytes=sum(r['bytes'] for r in results),
            compressed_bytes=plan['compressed_payload_bytes'], pairing_table=plan['pairing_table'],
            images_decoded=0, enhancement_CNN=0, GPU=False)); complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc())
        receipts = [json.loads(path.read_text()) for path in (cache/'receipts').glob('*.json')]
        write(out/'partial-manifest.json', dict(completed=False, successful_receipts=receipts,
              scope='Only individually completed verified PNGs; never claim full acquisition or decode.'))
        raise
    finally:
        write(out/'runtime.json', dict(completed_normally=complete, collected_members=len(results), images_decoded=0,
            enhancement_CNN=0, GPU=False, seconds=time.perf_counter()-start, ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__': main()
