"""Freeze official REDS validation metadata and prescribed members, without pixels."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
from urllib.request import urlopen
import zipfile

from reds_archive_r71 import RemoteZip

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'research/4.temporal-consistency'
OUT = BASE / 'process/exploration-r71/catalog001'
ASSETS = ROOT / '.cache/datasets/reds/r71/catalog001'
REVISION = '62dc25d16e6f43d2214f1b365023abda86f7a0ae'
PAGE = 'https://seungjunnah.github.io/Datasets/reds.html'


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    ASSETS.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    record = dict(started=datetime.now().astimezone().isoformat(), completed_normally=False,
                  decoded_pixels=0, image_payload_members_read=0)
    remote = None
    try:
        with urlopen(PAGE, timeout=30) as response:
            page = response.read(256 * 1024 + 1)
        assert len(page) <= 256 * 1024
        assert b'huggingface.co/datasets/snah/REDS/resolve/main/val_sharp.zip' in page
        (ASSETS / 'official-page.html').write_bytes(page)
        with urlopen('https://huggingface.co/api/datasets/snah/REDS?blobs=true', timeout=30) as response:
            raw = response.read(1024 * 1024 + 1)
        assert len(raw) <= 1024 * 1024
        metadata = json.loads(raw)
        assert metadata['sha'] == REVISION and metadata['id'] == 'snah/REDS'
        (ASSETS / 'hf-metadata.json').write_bytes(raw)
        member, = [r for r in metadata['siblings'] if r['rfilename'] == 'val_sharp.zip']
        assert member['size'] == member['lfs']['size']
        url = f'https://huggingface.co/datasets/snah/REDS/resolve/{REVISION}/val_sharp.zip'
        source = dict(official_page=PAGE, official_page_sha256=hashlib.sha256(page).hexdigest(),
                      repository='snah/REDS', revision=REVISION, metadata_sha256=hashlib.sha256(raw).hexdigest(),
                      archive_url=url, archive_bytes=member['size'], archive_lfs_sha256=member['lfs']['sha256'],
                      whole_archive_sha256_locally_verified=False, license='CC BY 4.0',
                      scope='Author-linked sharp validation, metadata only; not an official low-light benchmark.')
        write(OUT / 'source.json', source)
        remote = RemoteZip(url, member['size'])
        with zipfile.ZipFile(remote) as archive:
            rows = [dict(name=i.filename, size=i.file_size, compressed_size=i.compress_size,
                         crc32=f'{i.CRC:08x}', local_offset=i.header_offset,
                         method=i.compress_type, flags=i.flag_bits)
                    for i in archive.infolist() if not i.is_dir()]
        assert remote.transferred <= 4 * 1024 * 1024
        write(OUT / 'catalog.json', rows)
        assert all(r['name'].endswith('.png') for r in rows)
        scenes = sorted({Path(r['name']).parts[-2] for r in rows})
        assert scenes == [f'{i:03d}' for i in range(30)]
        for scene in scenes:
            names = sorted(Path(r['name']).name for r in rows if Path(r['name']).parts[-2] == scene)
            assert names == [f'{i:08d}.png' for i in range(100)]
        chosen = [scenes[i] for i in [0, 9, 19, 29]]
        selected = sorted([r for r in rows if Path(r['name']).parts[-2] in chosen
                           and int(Path(r['name']).stem) < 64], key=lambda r: r['name'])
        assert len(selected) == 256 and all(r['method'] in [0, 8] and not r['flags'] & 1 for r in selected)
        write(OUT / 'selection.json', dict(source=source, scenes=chosen, members=selected,
              expected_shape=[720, 1280, 3], role='DEV forever; custom low-light task not yet registered',
              selection_rule='Sorted validation scene indices 0/9/19/29, first64, before image payload',
              members_total=len(selected), decoded_pixels=0,
              expected_uncompressed_bytes=sum(r['size'] for r in selected),
              expected_compressed_bytes=sum(r['compressed_size'] for r in selected)))
        record.update(completed_normally=True, catalog_entries=len(rows), scenes=len(scenes),
                      selected_scenes=chosen, selected_members=len(selected))
    except Exception as error:
        record.update(error_type=type(error).__name__, error_status=getattr(error, 'code', None))
        raise
    finally:
        record.update(ended=datetime.now().astimezone().isoformat(), seconds=time.perf_counter() - started,
                      metadata_range_bytes=remote.transferred if remote else 0)
        if remote:
            write(OUT / 'range-receipts.json', dict(requests=remote.ranges, retries=remote.retries))
        write(OUT / 'runtime.json', record)
    print(json.dumps(record))


if __name__ == '__main__':
    main()
