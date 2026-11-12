"""Inspect official DRV raw ZIP directories only; never read an image member."""
import argparse
import json
import re
import time
import traceback
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path, PurePosixPath

from dynamic_training_r34 import BASE, ROOT, digest
from inspect_davis_external import RemoteZip

URL_ROOT = 'https://storage.googleapis.com/isl-datasets/DRV/'


class MetadataZip(RemoteZip):
    def read(self, count=-1):
        requested = min(self.size-self.position, count if count >= 0 else self.size-self.position)
        assert self.transferred+requested <= 4*1024*1024, 'Metadata-only per-archive transfer bound'
        return super().read(count)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--attempt', type=int, required=True)
    args = parser.parse_args(); assert args.attempt >= 2
    out = BASE/f'process/external-drv/archive-metadata-attempt{args.attempt:03d}'; out.mkdir(parents=True, exist_ok=False)
    cache = ROOT/f'.cache/datasets/DRV/archive-index/attempt{args.attempt:03d}'
    assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    cache.mkdir(parents=True, exist_ok=False)
    split = json.loads((BASE/'data/external-drv/official-split.json').read_text())
    ids = {v for row in split['splits'].values() for v in row['ids']}
    assert len(ids) == 202
    train = sorted(split['splits']['train']['ids'])
    chosen = [train[i] for i in [0, len(train)//3, 2*len(train)//3, len(train)-1]]
    start, complete, reports = time.perf_counter(), False, []
    try:
        for archive_name in ['short1.zip', 'short2.zip', 'short3.zip', 'short4.zip', 'short5.zip', 'long.zip']:
            remote = MetadataZip(URL_ROOT+archive_name)
            rows, matches = [], Counter()
            with zipfile.ZipFile(remote) as archive:
                for info in archive.infolist():
                    parts = PurePosixPath(info.filename).parts
                    # Frame basenames such as 0001.ARW can coincide with scene IDs.
                    # Prefer the nearest matching parent directory, not a frame number.
                    parents = [part for part in parts[:-1] if part in ids]
                    match = re.match(r'^(\d{4})_', parts[-1]) if parts else None
                    scene = parents[-1] if parents else (match.group(1) if match and match.group(1) in ids else None)
                    if scene: matches[scene] += not info.is_dir()
                    rows.append({'name': info.filename, 'size': info.file_size, 'compressed_size': info.compress_size,
                        'crc32': f'{info.CRC:08x}', 'offset': info.header_offset, 'method': info.compress_type,
                        'directory': info.is_dir(), 'matched_static_scene': scene})
            index = cache/(archive_name+'.index.json'); write(index, rows)
            report = {'at': datetime.now().astimezone().isoformat(), 'url': remote.url,
                'archive_bytes': remote.size, 'etag': remote.etag, 'last_modified': remote.last_modified,
                'transferred_directory_bytes': remote.transferred, 'members': len(rows),
                'extensions': dict(Counter(PurePosixPath(r['name']).suffix.lower() for r in rows if not r['directory'])),
                'matched_static_scene_counts': dict(sorted(matches.items())), 'future_train_candidates': chosen,
                'candidate_member_counts': {scene: matches[scene] for scene in chosen},
                'index_path': str(index.relative_to(ROOT)), 'index_sha256': digest(index),
                'image_members_read': 0, 'decoded_pixels': 0,
                'scope': 'ZIP64 central-directory metadata only. No raw member/header/pixel read, no scene quality selection.'}
            write(out/(archive_name+'.json'), report); reports.append(report)
            print(json.dumps({'archive': archive_name, 'members': len(rows),
                              'metadata_bytes': remote.transferred, 'candidate_counts': report['candidate_member_counts']}), flush=True)
        write(out/'summary.json', {'archives': reports, 'future_train_candidates': chosen,
              'total_metadata_bytes': sum(r['transferred_directory_bytes'] for r in reports),
              'image_members_read': 0, 'decoded_pixels': 0, 'new_CNN': 0, 'GPU_used': False})
        complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', {'completed_normally': complete, 'archives_completed': len(reports),
              'seconds': time.perf_counter()-start, 'ended': datetime.now().astimezone().isoformat()})


if __name__ == '__main__': main()
