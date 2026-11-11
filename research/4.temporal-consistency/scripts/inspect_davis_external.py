"""Read official DAVIS archive metadata/split text without reading image pixels."""
import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
from urllib.request import Request, urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[3]
URL = 'https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip'
OUT = ROOT/'research/4.temporal-consistency/process/external-davis'


class RemoteZip(io.RawIOBase):
    def __init__(self, url):
        self.url, self.position, self.transferred = url, 0, 0
        with urlopen(Request(url, method='HEAD'), timeout=30) as response:
            self.size = int(response.headers['Content-Length'])
            self.etag = response.headers['ETag']
            self.last_modified = response.headers['Last-Modified']

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position+offset if whence == 1 else self.size+offset
        assert 0 <= self.position <= self.size
        return self.position

    def tell(self):
        return self.position

    def read(self, count=-1):
        count = min(self.size-self.position, count if count >= 0 else self.size-self.position)
        if count == 0:
            return b''
        start, end = self.position, self.position+count-1
        def part(bounds):
            a, b = bounds
            request = Request(self.url, headers={'Range': f'bytes={a}-{b}', 'If-Match': self.etag})
            with urlopen(request, timeout=30) as response:
                assert response.status == 206
                assert response.headers['Content-Range'] == f'bytes {a}-{b}/{self.size}'
                value = response.read()
            assert len(value) == b-a+1
            return value
        bounds = [(a, min(a+256*1024, end+1)-1) for a in range(start, end+1, 256*1024)]
        if len(bounds) == 1:
            value = part(bounds[0])
        else:
            with ThreadPoolExecutor(max_workers=4) as pool:
                value = b''.join(pool.map(part, bounds))
        assert len(value) == count
        self.position += count
        self.transferred += count
        return value


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    remote = RemoteZip(URL)
    with zipfile.ZipFile(remote) as archive:
        names = archive.namelist()
        splits = {}
        for split in ['train', 'val']:
            name = next(n for n in names if n.endswith(f'/ImageSets/2017/{split}.txt'))
            value = archive.read(name)
            splits[split] = {'member': name, 'sha256': hashlib.sha256(value).hexdigest(),
                             'scenes': sorted(value.decode().split())}
        assert len(splits['train']['scenes']) == 60 and len(splits['val']['scenes']) == 30
        assert not set(splits['train']['scenes']) & set(splits['val']['scenes'])
        train = splits['train']['scenes']
        chosen = [train[i] for i in [0, len(train)//2, len(train)-1]]
        members = {}
        for scene in chosen:
            frames = sorted(n for n in names if '/JPEGImages/480p/'+scene+'/' in n and n.endswith('.jpg'))[:64]
            assert len(frames) >= 16
            members[scene] = [{'name': n, 'size': archive.getinfo(n).file_size,
                               'crc32': f'{archive.getinfo(n).CRC:08x}'} for n in frames]
    assert remote.transferred < 4*1024*1024
    report = {'observed_at_unix': time.time(), 'official_page': 'https://davischallenge.org/davis2017/code.html',
              'archive_url': URL, 'archive_bytes': remote.size, 'etag': remote.etag, 'last_modified': remote.last_modified,
              'transferred_metadata_bytes': remote.transferred, 'splits': splits,
              'selection': 'lexicographically sorted official TRAIN indices [0, n//2, n-1]; first min(64, available) frames',
              'pilot_members': members, 'decoded_image_pixels': 0, 'reserved_validation_image_members_read': 0,
              'scope': 'Video segmentation source metadata only; custom low-light degradation is not an official DAVIS enhancement benchmark.'}
    (OUT/'official-metadata.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'selected': chosen, 'frames': {s: len(v) for s, v in members.items()},
                      'metadata_bytes': remote.transferred, 'image_pixels_read': 0}))


if __name__ == '__main__':
    main()
