"""Range-read only central directories for the two official training archives."""
from datetime import datetime
import json
import time
import traceback

from catalog_crvd import BASE, PROTOCOL, ROOT, digest, mounted_data, write
from sdsd_archive_r67 import SplitArchive


def main():
    mounted_data()
    header = BASE/'process/external-crvd/head001/manifest.json'
    manifest = json.loads(header.read_text())
    assert manifest['completed'] and len(manifest['archives']) == 2
    out = BASE/'process/external-crvd/archive-catalog001'
    cache = ROOT/'.cache/datasets/CRVD/archive-catalog001'
    out.mkdir(exist_ok=False)
    cache.mkdir(exist_ok=False)
    write(out/'config.json', dict(at=datetime.now().astimezone().isoformat(),
          header_manifest_sha256=digest(header), protocol_sha256=digest(PROTOCOL),
          code_sha256={n: digest(BASE/'scripts'/n) for n in
                       ['catalog_crvd_archives.py', 'sdsd_archive_r67.py']},
          max_total_bytes=2*(65536+4*1024*1024), image_decodes=0))
    started, complete, count, transferred = time.perf_counter(), False, 0, 0
    try:
        for entry in manifest['archives']:
            assert entry['name'] == entry['role']+'_scene1.zip'
            volume = dict(disk=0, name=entry['name'], bytes=int(entry['content_length']),
                 url='https://drive.usercontent.google.com/download?id='+entry['id']+'&export=download&confirm=t')
            remote = SplitArchive([volume])
            try:
                rows, tail, central, info = remote.directory()
            finally:
                transferred += remote.transferred
            prefix = cache/entry['role']
            prefix.with_suffix('.tail').write_bytes(tail)
            prefix.with_suffix('.central').write_bytes(central)
            write(out/(entry['role']+'.json'), dict(source=entry, volume=volume, entries=rows,
                  directory=info, tail_path=str(prefix.with_suffix('.tail').relative_to(ROOT)),
                  central_path=str(prefix.with_suffix('.central').relative_to(ROOT)),
                  tail_sha256=digest(prefix.with_suffix('.tail')),
                  central_sha256=digest(prefix.with_suffix('.central')),
                  transferred_bytes=remote.transferred, technical_retries=remote.retries,
                  image_decodes=0, scope='Training scene1 central directory only; tail may include compressed image bytes.'))
            count += 1
            print(json.dumps(dict(archive=entry['role'], members=len(rows),
                                  transferred_bytes=remote.transferred)), flush=True)
        complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(out/'runtime.json', dict(completed_normally=complete, archives=count,
              archive_bytes_transferred=transferred, image_decodes=0, CNN=0, SGD=0, GPU=0,
              seconds=time.perf_counter()-started, ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__':
    main()
