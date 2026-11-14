"""Build the fixed official numerical core and owned float FFTW, no global install."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time
import traceback
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'research/4.temporal-consistency'
ASSET = ROOT/'.cache/tools/exploration-r70'
SOURCE = ROOT/'.cache/third_party/ipol-vbm3d/source001/src/vbm3d_1'


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path, value): path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    assert ASSET.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    source_record = json.loads((BASE/'process/exploration-r70/source-acquisition001.json').read_text())
    assert digest(ROOT/source_record['path']) == source_record['sha256']
    out = BASE/'process/exploration-r70/build001'; out.mkdir(exist_ok=False)
    work = ASSET/'build001'; work.mkdir(parents=True, exist_ok=False)
    prefix = work/'fftw-prefix'; download = work/'fftw-3.3.11.tar.gz'
    env = os.environ.copy(); env['MAKEFLAGS'] = ''; env.pop('DESTDIR', None)
    start, complete = time.perf_counter(), False
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in SOURCE.rglob('*') if p.is_file()}
    commands = []
    def run(command, cwd):
        commands.append(dict(command=[str(v) for v in command], cwd=str(cwd)))
        with (out/'build.log').open('ab') as log:
            log.write((json.dumps(commands[-1])+'\n').encode()); log.flush()
            subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    try:
        url = 'https://www.fftw.org/fftw-3.3.11.tar.gz'
        checksum_url = url+'.md5sum'
        with urllib.request.urlopen(checksum_url, timeout=30) as response:
            expected_md5 = response.read(4096).decode().split()[0]
        assert len(expected_md5) == 32 and all(c in '0123456789abcdefABCDEF' for c in expected_md5)
        with urllib.request.urlopen(url, timeout=30) as response, download.open('xb') as stream:
            assert response.status == 200
            total = 0
            while True:
                data = response.read(262144)
                if not data: break
                total += len(data); assert total <= 12*1024*1024; stream.write(data)
        actual_md5 = hashlib.md5(download.read_bytes()).hexdigest()
        assert actual_md5 == expected_md5.lower()
        write(out/'fftw-source.json', dict(at=datetime.now().astimezone().isoformat(), url=url,
              official_md5_url=checksum_url, official_md5=expected_md5, md5_match=True,
              path=str(download.relative_to(ROOT)), bytes=download.stat().st_size, sha256=digest(download)))
        with tarfile.open(download) as archive: archive.extractall(work/'fftw-source', filter='data')
        fftw_source = work/'fftw-source/fftw-3.3.11'
        run([str(fftw_source/'configure'), '--enable-float', '--enable-shared', '--disable-static',
             '--disable-fortran', '--enable-sse2', '--disable-dependency-tracking', f'--prefix={prefix}'], fftw_source)
        run(['make', '-j2'], fftw_source); run(['make', 'install'], fftw_source)
        objects = []
        units = [BASE/'scripts/vbm3d_array_r70.cpp', SOURCE/'src/vbm3d.cpp', SOURCE/'src/lib_transforms.cpp',
                 SOURCE/'src/Utilities/Utilities.cpp', SOURCE/'src/Utilities/LibVideoT.cpp', SOURCE/'src/Utilities/LibImages.cpp']
        flags = ['-O3', '-ftree-vectorize', '-funroll-loops', '-std=c++11', '-fopenmp',
                 '-ffunction-sections', '-fdata-sections', '-I'+str(prefix/'include'), '-I'+str(SOURCE/'src')]
        for ordinal, unit in enumerate(units):
            obj = work/f'unit{ordinal}.o'; objects.append(obj)
            run(['/usr/bin/g++', *flags, '-c', str(unit), '-o', str(obj)], work)
        binary = work/'vbm3d-array'
        run(['/usr/bin/g++', '-fopenmp', '-Wl,--gc-sections', *map(str, objects),
             '-L'+str(prefix/'lib'), '-Wl,-rpath,'+str(prefix/'lib'), '-lfftw3f', '-lm', '-o', str(binary)], work)
        dependencies = subprocess.check_output(['ldd', str(binary)], text=True)
        assert str(prefix/'lib') in dependencies and 'not found' not in dependencies
        assert all(digest(ROOT/name) == sha for name, sha in source_hashes.items())
        write(out/'completed.json', dict(at=datetime.now().astimezone().isoformat(), completed=True,
              binary=str(binary.relative_to(ROOT)), binary_sha256=digest(binary),
              FFTW_library=str((prefix/'lib/libfftw3f.so').relative_to(ROOT)),
              FFTW_library_sha256=digest(prefix/'lib/libfftw3f.so'), dependencies=dependencies,
              compiler=subprocess.check_output(['/usr/bin/g++', '--version'], text=True).splitlines()[0],
              commands=commands, original_source_sha256=source_hashes, original_sources_unmodified=True,
              adapter_source_sha256=digest(BASE/'scripts/vbm3d_array_r70.cpp'),
              script_sha256=digest(Path(__file__)),
              scope='Numerical core build only, original initializer/core/color path, no image I/O library or real/pseudo image execution.'))
        complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(completed_normally=complete, commands_attempted=commands,
              seconds=time.perf_counter()-start, new_real_images=0, new_CNN=0, new_SGD=0, new_GPU=0,
              ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__': main()
