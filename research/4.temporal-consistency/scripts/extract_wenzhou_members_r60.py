"""Extract an explicit short allowlist from a verified author archive; no codecs."""
import argparse
import ctypes as C
import ctypes.util
import json
import time
import traceback
from datetime import datetime
from pathlib import Path, PurePosixPath

from dynamic_training_r34 import BASE, ROOT, digest
from resume_wenzhou_samples_r52 import file_digest
from fetch_wenzhou_samples_r52 import write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    catalog_path = BASE/'process/external-night-wenzhou/catalog002/catalog.json'
    catalog = json.loads(catalog_path.read_text())
    source = ROOT/catalog['archive_manifest']
    assert digest(source) == catalog['manifest_sha256']
    manifest = json.loads(source.read_text()); assert manifest['download_complete']
    archive = ROOT/manifest['path']
    assert archive.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    assert archive.stat().st_size == manifest['bytes'] and file_digest(archive) == manifest['sha256']
    index = ROOT/catalog['index']; assert digest(index) == catalog['index_sha256']
    members = {row['name']: row for row in json.loads(index.read_text())}
    selected = {row['member']: row for row in plan['selection']}
    assert len(selected) == len(plan['selection']) and 0 < len(selected) <= 6
    assert sum(members[name]['size'] for name in selected) == plan['expected_bytes']
    assert plan['expected_bytes'] < 2_000_000_000
    cache, out = ROOT/plan['cache'], ROOT/plan['out']
    assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    assert out.resolve().is_relative_to((BASE/'process/external-night-wenzhou').resolve())
    cache.mkdir(parents=True, exist_ok=False); out.mkdir(exist_ok=False)
    lib = C.CDLL(ctypes.util.find_library('archive'))
    lib.archive_version_string.restype = C.c_char_p
    lib.archive_read_new.restype = C.c_void_p
    for name in ['archive_read_support_format_all', 'archive_read_support_filter_all',
                 'archive_read_data_skip', 'archive_read_free']:
        getattr(lib, name).argtypes = [C.c_void_p]; getattr(lib, name).restype = C.c_int
    lib.archive_read_open_filename.argtypes = [C.c_void_p, C.c_char_p, C.c_size_t]
    lib.archive_read_next_header.argtypes = [C.c_void_p, C.POINTER(C.c_void_p)]
    lib.archive_entry_size.argtypes = [C.c_void_p]; lib.archive_entry_size.restype = C.c_int64
    lib.archive_entry_filetype.argtypes = [C.c_void_p]; lib.archive_entry_filetype.restype = C.c_uint
    for name in ['archive_error_string', 'archive_entry_pathname', 'archive_entry_symlink', 'archive_entry_hardlink']:
        getattr(lib, name).argtypes = [C.c_void_p]; getattr(lib, name).restype = C.c_char_p
    lib.archive_read_data.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t]
    lib.archive_read_data.restype = C.c_ssize_t
    libraries = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'libarchive.so.' in line})
    assert libraries and all(p.startswith(('/usr/lib/', '/lib/')) for p in libraries)
    write(out/'config.json', dict(at=datetime.now().astimezone().isoformat(), plan=plan,
          plan_sha256=digest(args.plan), catalog_sha256=digest(catalog_path), archive_sha256=manifest['sha256'],
          source_sha256=digest(Path(__file__)), libarchive=lib.archive_version_string().decode(),
          system_library_paths=libraries, codecs_called=False, CNN=0, GPU=False))
    start, complete, handle, rows = time.perf_counter(), False, None, []
    try:
        handle = lib.archive_read_new(); assert handle
        def check(status):
            if status != 0:
                raise RuntimeError((lib.archive_error_string(handle) or b'libarchive error').decode('utf8', 'replace'))
        check(lib.archive_read_support_format_all(handle)); check(lib.archive_read_support_filter_all(handle))
        check(lib.archive_read_open_filename(handle, str(archive).encode(), 10240))
        entry, buffer = C.c_void_p(), C.create_string_buffer(1024*1024)
        while True:
            status = lib.archive_read_next_header(handle, C.byref(entry))
            if status == 1: break
            check(status)
            raw = lib.archive_entry_pathname(entry); name = raw.decode('utf8')
            if name not in selected:
                check(lib.archive_read_data_skip(handle)); continue
            meta, choice = members[name], selected[name]
            assert raw.hex() == meta['name_bytes_hex']
            assert lib.archive_entry_filetype(entry) == 0o100000 and lib.archive_entry_size(entry) == meta['size']
            assert not lib.archive_entry_symlink(entry) and not lib.archive_entry_hardlink(entry)
            dest = choice['destination']
            assert PurePosixPath(dest).name == dest and dest not in ['.', '..']
            target, count = cache/dest, 0
            with target.open('xb') as stream:
                while True:
                    size = lib.archive_read_data(handle, buffer, len(buffer))
                    assert size >= 0, 'Archive member read/CRC failed.'
                    if not size: break
                    count += size; assert count <= meta['size']; stream.write(buffer.raw[:size])
            assert count == meta['size']
            row = dict(member=name, path=str(target.relative_to(ROOT)), bytes=count, sha256=file_digest(target))
            rows.append(row)
            print(json.dumps(row), flush=True)
        assert len(rows) == len(selected) and sum(r['bytes'] for r in rows) == plan['expected_bytes']
        write(out/'manifest.json', dict(at=datetime.now().astimezone().isoformat(), files=rows,
              video_frames_returned=0, codecs_called=False, CNN=0, GPU=False))
        complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        if handle: lib.archive_read_free(handle)
        write(out/'runtime.json', dict(completed_normally=complete, members_extracted=len(rows),
              bytes_extracted=sum(r['bytes'] for r in rows), codecs_called=False,
              seconds=time.perf_counter()-start, ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__': main()
