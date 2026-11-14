"""Inspect the official real-phone archive directory; do not read image members."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen
import zipfile

from inspect_davis_external import ROOT, RemoteZip

URL = 'https://drive.usercontent.google.com/download?id=1QS4FgT5aTQNYy-eHZ_A89rLoZgx_iysR&export=download&confirm=t'
BASE = ROOT/'research/4.temporal-consistency'


class RangeOnlyZip(RemoteZip):
    def __init__(self, url):
        self.url, self.position, self.transferred = url, 0, 0
        with urlopen(Request(url, headers={'Range': 'bytes=0-0'}), timeout=30) as response:
            assert response.status == 206
            self.size = int(response.headers['Content-Range'].split('/')[-1])
            self.etag = response.headers.get('ETag')
            self.last_modified = response.headers.get('Last-Modified')
            assert len(response.read()) == 1
        self.transferred = 1


def main():
    remote = RangeOnlyZip(URL)
    with zipfile.ZipFile(remote) as archive:
        infos = archive.infolist()
        rows = [{'name': i.filename, 'size': i.file_size, 'compressed_size': i.compress_size,
                 'method': i.compress_type, 'crc32': f'{i.CRC:08x}', 'offset': i.header_offset}
                for i in infos]
        image_rows = [r for r in rows if Path(r['name']).suffix.lower() in ['.jpg', '.jpeg', '.png'] and not r['name'].startswith('__MACOSX/')]
        folders = Counter(str(Path(r['name']).parent) for r in image_rows)
    assert remote.transferred < 32*1024*1024
    cache = ROOT/'.cache/datasets/LLIV-Phone/zip-index.json'
    assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({'members': rows, 'central_directory_start': archive.start_dir}, ensure_ascii=False, indent=2)+'\n')
    report = {'observed_at_unix': time.time(), 'official_file_id': '1QS4FgT5aTQNYy-eHZ_A89rLoZgx_iysR',
              'official_repository_commit': '106e914a5fa5842cca454eb0a48076e8b274f85f',
              'archive_bytes': remote.size, 'etag': remote.etag, 'last_modified': remote.last_modified,
              'transferred_metadata_bytes': remote.transferred, 'members': len(rows), 'image_members': len(image_rows),
              'image_directories': dict(sorted(folders.items())), 'member_extension_counts': dict(Counter(Path(r['name']).suffix.lower() for r in rows)),
              'cached_index': str(cache.relative_to(ROOT)), 'index_sha256': hashlib.sha256(cache.read_bytes()).hexdigest(),
              'image_members_read': 0, 'pixels_decoded': 0,
              'scope': 'Official archive metadata only; directory names are not proof of paired ground truth, temporal cadence or device calibration.'}
    out = BASE/'process/external-phone/archive-metadata.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'archive_bytes': remote.size, 'images': len(image_rows), 'image_directories': len(folders),
                      'metadata_bytes': remote.transferred, 'pixels_decoded': 0}), flush=True)


if __name__ == '__main__':
    main()
