"""Fetch only preregistered TRAIN JPEG members of the official DAVIS archive."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import time
from urllib.request import Request, urlopen
import zipfile
import zlib

from inspect_davis_external import ROOT, RemoteZip

BASE = ROOT/'research/4.temporal-consistency'


def fetch(item):
    record, url, total, etag, offset, end, compressed, method = item
    target = ROOT/'.cache/datasets/DAVIS2017-pilot'/record['name'].removeprefix('DAVIS/')
    assert target.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        value = target.read_bytes()
    else:
        for attempt in range(4):
            try:
                request = Request(url, headers={'Range': f'bytes={offset}-{end}', 'If-Match': etag})
                with urlopen(request, timeout=30) as response:
                    assert response.status == 206
                    assert response.headers['Content-Range'] == f'bytes {offset}-{end}/{total}'
                    block = response.read()
                assert len(block) == end-offset+1
                assert block[:4] == b'PK\x03\x04'
                flags, local_method = struct.unpack_from('<HH', block, 6)
                assert not flags & 1 and local_method == method
                namesize, extrasize = struct.unpack_from('<HH', block, 26)
                name = block[30:30+namesize].decode('utf8')
                assert name == record['name']
                start = 30+namesize+extrasize
                packed = block[start:start+compressed]
                assert len(packed) == compressed
                value = packed if method == zipfile.ZIP_STORED else zlib.decompress(packed, -zlib.MAX_WBITS)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(attempt+1)
    assert len(value) == record['size']
    assert f'{zlib.crc32(value) & 0xffffffff:08x}' == record['crc32']
    if not target.exists():
        temporary = target.with_suffix('.jpg.partial')
        temporary.write_bytes(value)
        temporary.replace(target)
    print(json.dumps({'member': record['name'], 'bytes': len(value), 'CRC_verified': True}), flush=True)
    return {**record, 'path': str(target.relative_to(ROOT)), 'sha256': hashlib.sha256(value).hexdigest()}


def main():
    protocol = BASE/'docs/exploration-r30-protocol.md'
    metadata_path = BASE/'process/external-davis/official-metadata.json'
    out = BASE/'data/external-davis/manifest.json'
    assert protocol.is_file() and not out.exists()
    metadata = json.loads(metadata_path.read_text())
    remote = RemoteZip(metadata['archive_url'])
    assert remote.size == metadata['archive_bytes'] and remote.etag == metadata['etag']
    with zipfile.ZipFile(remote) as archive:
        infos = archive.infolist()
        offsets = sorted([i.header_offset for i in infos]+[archive.start_dir])
        end_offsets = dict(zip(offsets, offsets[1:]))
        work = []
        for scene, members in metadata['pilot_members'].items():
            assert scene in metadata['splits']['train']['scenes']
            for record in members:
                info = archive.getinfo(record['name'])
                assert info.file_size == record['size'] and f'{info.CRC:08x}' == record['crc32']
                assert info.compress_type in [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED]
                end = end_offsets[info.header_offset]-1
                assert end-info.header_offset+1 <= info.compress_size+128*1024
                work.append((record, remote.url, remote.size, remote.etag, info.header_offset, end, info.compress_size, info.compress_type))
    with ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(fetch, work))
    assert len(values) == 192
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {'archive_url': remote.url, 'archive_bytes': remote.size, 'etag': remote.etag, 'frames': values,
              'protocol_sha256': hashlib.sha256(protocol.read_bytes()).hexdigest(),
              'metadata_sha256': hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
              'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'decoded_image_pixels': 0, 'validation_image_members_read': 0,
              'scope': 'Only registered TRAIN JPEG bytes fetched and ZIP CRC-verified; no images decoded yet.'}
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
