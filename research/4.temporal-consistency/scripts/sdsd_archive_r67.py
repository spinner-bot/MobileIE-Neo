"""Bounded author-linked multi-volume ZIP ranges; no image decoding."""
import hashlib
import http.client
import io
import json
from pathlib import Path
import re
import struct
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'research/4.temporal-consistency'
LISTINGS = ROOT/'research/5.prior-art-survey/process/data-readiness-r16'
MAX_RANGE = 1024*1024


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value): path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def volumes_for(domain):
    source = LISTINGS/f'{domain}_png-listing.json'
    items = json.loads(source.read_text())['children']
    parts = sorted([r for r in items if re.search(r'\.z\d+$', r['name'])], key=lambda r: int(r['name'].rsplit('z', 1)[1]))
    last, = [r for r in items if r['name'] == domain+'.zip']
    assert [int(r['name'].rsplit('z', 1)[1]) for r in parts] == list(range(1, len(parts)+1))
    return [dict(disk=i, name=row['name'], bytes=row['bytes'],
                 url='https://drive.usercontent.google.com/download?id='+row['id']+'&export=download&confirm=t')
            for i, row in enumerate(parts+[last])]


class SplitArchive:
    def __init__(self, volumes):
        self.volumes = volumes; self.transferred = 0; self.retries = []; self.observed_etags = {}

    def read_exact(self, disk, start, count):
        volume = self.volumes[disk]
        assert 0 <= start < volume['bytes'] and 0 < count <= min(MAX_RANGE, volume['bytes']-start)
        end = start+count-1
        for attempt in range(1, 4):
            try:
                request = Request(volume['url'], headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'})
                with urlopen(request, timeout=60) as response:
                    assert response.status == 206, f'Expected206, got{response.status}; no quota/auth bypass or retry.'
                    assert response.headers['Content-Range'] == f'bytes {start}-{end}/{volume["bytes"]}'
                    assert int(response.headers['Content-Length']) == count
                    etag = response.headers.get('ETag')
                    if etag is not None:
                        assert self.observed_etags.setdefault(disk, etag) == etag
                    value = response.read(count+1)
                assert len(value) == count
                self.transferred += count
                return value
            except (URLError, TimeoutError, ConnectionError, http.client.IncompleteRead) as error:
                if isinstance(error, HTTPError) and error.code < 500: raise
                self.retries.append(dict(disk=disk, start=start, count=count, attempt=attempt, error=repr(error)))
                if attempt == 3: raise
                time.sleep(attempt)

    def read_span(self, disk, offset, count):
        pieces = []
        while count:
            if offset == self.volumes[disk]['bytes']: disk, offset = disk+1, 0
            assert disk < len(self.volumes)
            take = min(count, MAX_RANGE, self.volumes[disk]['bytes']-offset)
            pieces.append(self.read_exact(disk, offset, take)); offset += take; count -= take
        return b''.join(pieces)

    def directory(self):
        disk = len(self.volumes)-1; size = self.volumes[-1]['bytes']
        tail = self.read_span(disk, size-65536, 65536)
        position = tail.rfind(b'PK\x05\x06'); assert position >= 0
        fields = struct.unpack('<4s4H2IH', tail[position:position+22])
        _, end_disk, directory_disk, disk_entries, entries, directory_size, directory_offset, comment = fields
        assert end_disk == directory_disk == disk and disk_entries == entries < 65535
        assert position+22+comment == len(tail) and 0 < directory_size <= 4*1024*1024
        assert directory_offset+directory_size <= size-22-comment
        central = self.read_span(disk, directory_offset, directory_size)
        # This surrogate is solely a central-directory parser. Member data is never opened here.
        single_end = struct.pack('<4s4H2IH', b'PK\x05\x06', 0, 0, entries, entries, directory_size, 0, 0)
        with zipfile.ZipFile(io.BytesIO(central+single_end)) as archive:
            rows = [dict(name=info.filename, disk=info.volume, local_offset=info.header_offset,
                size=info.file_size, compressed_size=info.compress_size, crc32=f'{info.CRC:08x}',
                method=info.compress_type, flags=info.flag_bits, directory=info.is_dir()) for info in archive.infolist()]
        assert len(rows) == entries and all(0 <= r['disk'] < len(self.volumes) for r in rows)
        return rows, tail, central, dict(end_disk=end_disk, directory_disk=directory_disk, entries=entries,
              directory_bytes=directory_size, directory_offset=directory_offset,
              scope='Original disk/local offsets preserved, ZIP64 entry extras decoded by standard zipfile; surrogate never used to open payload.')
