"""Fetch four frozen training TIFFs, verify ZIP identity and native RAW format."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import struct
import sys
import time
import traceback
from urllib.request import Request, urlopen
import zlib

import cv2
import numpy as np

from catalog_crvd import ASSETS, BASE, PROTOCOL, ROOT, digest, mounted_data, write
from sdsd_archive_r67 import MAX_RANGE, SplitArchive


def native_tiff_tags(data):
    assert data[:2] in [b'II', b'MM']
    endian = '<' if data[:2] == b'II' else '>'
    assert struct.unpack_from(endian+'H', data, 2)[0] == 42
    position = struct.unpack_from(endian+'I', data, 4)[0]
    count = struct.unpack_from(endian+'H', data, position)[0]
    assert 0 < count <= 256 and position+2+12*count+4 <= len(data)
    wanted, tags = {256, 257, 258, 259, 277, 339}, {}
    for i in range(count):
        start = position+2+i*12
        tag, kind, number = struct.unpack_from(endian+'HHI', data, start)
        if tag in wanted:
            assert number == 1 and kind in [3, 4]
            tags[str(tag)] = struct.unpack_from(endian+('H' if kind == 3 else 'I'), data, start+8)[0]
    assert tags['256'] == 1920 and tags['257'] == 1080 and tags.get('277', 1) == 1
    return tags


class SingleAttemptArchive(SplitArchive):
    def __init__(self, volume, budget):
        super().__init__([volume])
        self.budget = budget

    def read_exact(self, disk, start, count):
        volume = self.volumes[disk]
        assert disk == 0 and 0 < count <= MAX_RANGE and 0 <= start < volume['bytes']
        assert start+count <= volume['bytes'] and self.budget['used']+count <= self.budget['limit']
        end = start+count-1
        request = Request(volume['url'], headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'})
        with urlopen(request, timeout=30) as response:
            assert response.status == 206
            assert response.headers.get('Content-Range') == f'bytes {start}-{end}/{volume["bytes"]}'
            assert int(response.headers['Content-Length']) == count
            value = response.read(count+1)
        assert len(value) == count
        self.transferred += count
        self.budget['used'] += count
        return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    assert args.attempt > 0
    mounted_data()
    cv2.setNumThreads(1)
    selection = BASE/'data/external-crvd/packet-selection001.json'
    plan = json.loads(selection.read_text())
    assert digest(PROTOCOL) == plan['protocol_sha256'] and len(plan['members']) == 4
    assert all(digest(ROOT/p) == value for p, value in plan['source_sha256'].items())
    out = BASE/f'process/external-crvd/acquisition{args.attempt:03d}'
    cache = ROOT/f'.cache/datasets/CRVD/qualification{args.attempt:03d}'
    assert cache.resolve().is_relative_to(ASSETS.resolve())
    out.mkdir(exist_ok=False)
    cache.mkdir(exist_ok=False)
    budget = dict(used=0, limit=plan['max_transfer_bytes'])
    write(out/'config.json', dict(at=datetime.now().astimezone().isoformat(),
          selection_sha256=digest(selection), protocol_sha256=digest(PROTOCOL),
          code_sha256=digest(Path(__file__)), max_transfer_bytes=budget['limit'],
          technical_retries=0, decodes='Native TIFF only; no clipping/casting/rendering'))
    started, complete, results = time.perf_counter(), False, []
    try:
        for row in plan['members']:
            remote = SingleAttemptArchive(row['volume'], budget)
            header = remote.read_span(0, row['local_offset'], 1024)
            fields = struct.unpack_from('<4s5H3I2H', header)
            assert fields[0] == b'PK\x03\x04' and not fields[2] & 1 and fields[3] == row['method']
            name_size, extra_size = fields[-2:]
            begin = 30+name_size+extra_size
            assert begin <= len(header)
            assert header[30:30+name_size].decode('utf-8' if fields[2] & 2048 else 'cp437') == row['name']
            compressed = cache/(row['role']+'.deflate')
            partial = compressed.with_suffix('.deflate.part')
            with partial.open('xb') as stream:
                for offset in range(0, row['compressed_size'], MAX_RANGE):
                    stream.write(remote.read_span(0, row['local_offset']+begin+offset,
                                     min(MAX_RANGE, row['compressed_size']-offset)))
            assert partial.stat().st_size == row['compressed_size']
            partial.rename(compressed)
            payload = compressed.read_bytes()
            if row['method'] == 8:
                decoder = zlib.decompressobj(-zlib.MAX_WBITS)
                data = decoder.decompress(payload, plan['max_member_bytes']+1)
                assert decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail
            else:
                data = payload
            assert len(data) == row['size'] <= plan['max_member_bytes']
            assert f'{zlib.crc32(data):08x}' == row['crc32']
            path = cache/row['output']
            with path.open('xb') as stream:
                stream.write(data)
            tags = native_tiff_tags(data)
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            assert image is not None and list(image.shape) == plan['expected_shape']
            assert np.isfinite(image).all()
            sample, bits = tags.get('339', 1), tags['258']
            expected = {(1, 16): 'uint16', (1, 32): 'uint32', (3, 32): 'float32'}
            assert str(image.dtype) == expected[sample, bits], 'Decoder must preserve native TIFF type'
            if row['role'] != 'bm3d_reference':
                assert image.dtype == np.uint16 and int(image.max()) <= plan['white_level']
            record = dict(role=row['role'], member=row['name'], source_archive=row['archive'],
                  path=str(path.relative_to(ROOT)), sha256=digest(path), bytes=len(data), crc32=row['crc32'],
                  compressed_path=str(compressed.relative_to(ROOT)), compressed_sha256=digest(compressed),
                  transferred_bytes=remote.transferred, shape=list(image.shape), dtype=str(image.dtype),
                  minimum=float(image.min()), maximum=float(image.max()), native_tags=tags,
                  reference_type={'bm3d_reference': 'Average plus BM3D',
                                  'averaged_reference': 'Original multi-capture average'}.get(row['role'], 'Noisy sensor repetition'),
                  rendered=False, clipped=False, cast=False)
            write(out/(row['role']+'.json'), record)
            results.append(record)
            print(json.dumps(record), flush=True)
        assert 'torch' not in sys.modules
        write(out/'manifest.json', dict(completed=True, files=results,
              selection_sha256=digest(selection), source_documented_Bayer=plan['source_documented_Bayer'],
              scope='Four native training TIFFs only; no noise calibration, benchmark, model or independent GT claim.'))
        complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(out/'runtime.json', dict(completed_normally=complete, qualified_files=len(results),
              successful_range_body_bytes=budget['used'], image_decodes=len(results),
              CNN=0, SGD=0, GPU=0, seconds=time.perf_counter()-started,
              ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__':
    main()
