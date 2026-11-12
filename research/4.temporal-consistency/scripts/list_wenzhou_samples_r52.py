"""List the acquired RAR using the OS libarchive, without extracting or decoding."""
import ctypes as C
import ctypes.util
import json
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path, PurePosixPath

from dynamic_training_r34 import BASE, ROOT, digest


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    source = BASE/'process/external-night-wenzhou/acquisition001/manifest.json'
    manifest = json.loads(source.read_text()); assert manifest['download_complete']
    path = ROOT/manifest['path']
    assert path.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    assert path.stat().st_size == manifest['bytes'] and digest(path) == manifest['sha256']
    out = BASE/'process/external-night-wenzhou/catalog001'; out.mkdir(exist_ok=False)
    start, complete, handle, rows = time.perf_counter(), False, None, []
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
    for name in ['archive_error_string', 'archive_entry_pathname', 'archive_entry_pathname_utf8',
                 'archive_entry_symlink', 'archive_entry_hardlink']:
        getattr(lib, name).argtypes = [C.c_void_p]; getattr(lib, name).restype = C.c_char_p
    loaded = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                     if 'libarchive.so.' in line})
    assert loaded and all(p.startswith(('/usr/lib/', '/lib/')) for p in loaded)
    try:
        handle = lib.archive_read_new(); assert handle
        assert lib.archive_read_support_format_all(handle) == 0
        assert lib.archive_read_support_filter_all(handle) == 0
        def check(status):
            if status != 0:
                error = lib.archive_error_string(handle)
                raise RuntimeError(f'libarchive status {status}: '+(error or b'').decode('utf8', 'replace'))
        check(lib.archive_read_open_filename(handle, str(path).encode(), 10240))
        entry = C.c_void_p()
        while True:
            status = lib.archive_read_next_header(handle, C.byref(entry))
            if status == 1: break
            check(status)
            raw = lib.archive_entry_pathname(entry)
            utf8 = lib.archive_entry_pathname_utf8(entry)
            name = (utf8 or raw).decode('utf8', 'replace')
            rows.append({'name': name, 'name_bytes_hex': raw.hex(), 'size': int(lib.archive_entry_size(entry)),
                         'file_type_octal': oct(lib.archive_entry_filetype(entry)),
                         'symlink': bool(lib.archive_entry_symlink(entry)),
                         'hardlink': bool(lib.archive_entry_hardlink(entry))})
            assert len(rows) <= 10000
            check(lib.archive_read_data_skip(handle))
        index = path.parent/'catalog001.json'; write(index, rows)
        write(out/'catalog.json', {'at': datetime.now().astimezone().isoformat(),
              'archive_manifest': str(source.relative_to(ROOT)), 'manifest_sha256': digest(source),
              'index': str(index.relative_to(ROOT)), 'index_sha256': digest(index), 'members': len(rows),
              'extension_counts': dict(Counter(PurePosixPath(r['name']).suffix.lower() for r in rows)),
              'video_members': [r for r in rows if PurePosixPath(r['name']).suffix.lower() in
                               ['.mp4', '.mov', '.avi', '.mkv', '.m4v']],
              'libarchive': lib.archive_version_string().decode(), 'system_library_paths': loaded,
              'explicit_members_extracted': 0, 'video_frames_decoded': 0,
              'scope': 'Headers/listing only. Solid archive skipping may decompress bytes internally; no image/video codec or candidate inference.'})
        complete = True
        print(json.dumps({'members': len(rows), 'video_frames_decoded': 0}))
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        if handle: lib.archive_read_free(handle)
        write(out/'runtime.json', {'completed_normally': complete, 'members_listed': len(rows),
              'seconds': time.perf_counter()-start, 'ended': datetime.now().astimezone().isoformat()})


if __name__ == '__main__': main()
