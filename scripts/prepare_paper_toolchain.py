"""Install one pinned official typesetter inside the project's owned asset cache."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

from project_assets import ASSETS, REPO

VERSION = '0.17.0'
NAME = 'tectonic-0.17.0-x86_64-unknown-linux-musl.tar.gz'
URL = 'https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/'+NAME
SIZE = 10151914
SHA256 = '8533d07f9ccbd7a65824b9e0459041bca34af1eb33daba48f59215593753a3b7'


def main():
    cache = ASSETS/'cache/tools/tectonic'/VERSION
    assert cache.resolve().is_relative_to(ASSETS.resolve())
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache/NAME
    if not archive.exists():
        partial = cache/(NAME+'.part-'+datetime.now().strftime('%Y%m%dT%H%M%S'))
        request = urllib.request.Request(URL, headers={'User-Agent': 'MobileIE-Neo research typesetting'})
        with urllib.request.urlopen(request, timeout=60) as response, partial.open('xb') as output:
            while block := response.read(1024*1024):
                output.write(block)
        assert partial.stat().st_size == SIZE
        assert hashlib.sha256(partial.read_bytes()).hexdigest() == SHA256
        partial.rename(archive)
    assert archive.stat().st_size == SIZE and hashlib.sha256(archive.read_bytes()).hexdigest() == SHA256
    binary = cache/'tectonic'
    if not binary.exists():
        with tarfile.open(archive) as package:
            matches = [m for m in package.getmembers() if m.isfile() and Path(m.name).name == 'tectonic']
            assert len(matches) == 1
            content = package.extractfile(matches[0]).read()
        binary.write_bytes(content)
        binary.chmod(0o755)
    record = {'prepared_at': datetime.now().astimezone().isoformat(), 'version': VERSION,
              'source': 'Official tectonic-typesetting/tectonic GitHub release, API-published asset digest',
              'release_url': 'https://github.com/tectonic-typesetting/tectonic/releases/tag/tectonic%400.17.0',
              'archive_url': URL, 'archive_bytes': SIZE, 'archive_sha256': SHA256,
              'binary': str(binary), 'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
              'scope': 'Project-owned binary/cache only; no global installation, borrowed environment or shell startup changes.'}
    target = REPO/'paper/process/toolchain.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    assert not target.exists(), 'Preserve the prior toolchain record'
    target.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'version': VERSION, 'binary': str(binary), 'archive_verified': True}))


if __name__ == '__main__':
    main()
