"""Install hash-verified extra wheels into an owned tool target, leaving live venv alone."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from html.parser import HTMLParser
from urllib.parse import unquote, urljoin, urlsplit, parse_qs
from urllib.request import urlopen

from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'research/4.temporal-consistency'
sys.path.insert(0, str(ROOT/'scripts'))
from project_assets import mounted_data
from prepare_owned_wheelhouse import download_part

PACKAGES = [('cvvdp', '0.5.7'), ('huggingface-hub', '0.33.4'), ('imageio', '2.37.0'),
            ('ffmpeg-python', '0.2.0'), ('future', '1.0.0'), ('requests', '2.32.5'),
            ('urllib3', '2.5.0'), ('idna', '3.11'), ('charset-normalizer', '3.4.4'), ('certifi', '2025.11.12')]
TARGET = ROOT/'.cache/tools/exploration-r66/cvvdp057'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


class Links(HTMLParser):
    def __init__(self): super().__init__(); self.links = []
    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            for name, value in attrs:
                if name == 'href': self.links.append(value)


def main():
    mounted_data()
    assert TARGET.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    out = BASE/'process/exploration-r66/package001'; out.mkdir(parents=True, exist_ok=False)
    wheels = ROOT/'.cache/tools/exploration-r66/wheels'; wheels.mkdir(parents=True, exist_ok=False)
    assert not TARGET.exists()
    start, complete, records = time.perf_counter(), False, []
    try:
        tags = {tag: rank for rank, tag in enumerate(sys_tags())}
        for name, version in PACKAGES:
            metadata_url = f'https://pypi.org/pypi/{name}/{version}/json'
            with urlopen(metadata_url, timeout=30) as response: metadata = json.load(response)
            candidates = []
            for item in metadata['urls']:
                if not item['filename'].endswith('.whl'): continue
                _, _, _, supported = parse_wheel_filename(item['filename'])
                rank = min((tags[tag] for tag in supported if tag in tags), default=None)
                if rank is not None: candidates.append((rank, item))
            assert candidates, (name, version)
            item = min(candidates, key=lambda row: row[0])[1]
            records.append(dict(package=name, version=version, filename=item['filename'], url=item['url'],
                                bytes=item['size'], sha256=item['digests']['sha256'], metadata_url=metadata_url))
        index = 'https://download.pytorch.org/whl/cu126/torchvision/'
        with urlopen(index, timeout=30) as response: content = response.read().decode()
        parser = Links(); parser.feed(content)
        choices = [urljoin(index, link) for link in parser.links
                   if 'torchvision-0.24.1+cu126-cp310-cp310-' in unquote(link)
                   and 'x86_64.whl' in link and 'linux' in link]
        assert len(choices) == 1, choices
        url = choices[0]; parsed = urlsplit(url); sha = parse_qs(parsed.fragment)['sha256'][0]
        url = url.split('#', 1)[0]
        with urlopen(url, timeout=30) as response:
            size = int(response.headers['Content-Length'])
        records.append(dict(package='torchvision', version='0.24.1+cu126', filename=unquote(Path(parsed.path).name),
                            url=url, bytes=size, sha256=sha, metadata_url=index))
        write(out/'sources-before-download.json', dict(at=datetime.now().astimezone().isoformat(),
              wheels=records, target=str(TARGET.relative_to(ROOT)), source='Official PyPI files and PyTorch cu126 only.'))
        for item in records:
            assert item['url'].startswith(('https://files.pythonhosted.org/', 'https://download.pytorch.org/'))
            path = wheels/item['filename']; partial = path.with_suffix('.whl.partial')
            chunk, size = 256*1024, item['bytes']
            with partial.open('xb') as stream, ThreadPoolExecutor(max_workers=4) as pool:
                stream.truncate(size)
                tasks = [pool.submit(download_part, item['url'], offset, min(offset+chunk, size)-1, size)
                         for offset in range(0, size, chunk)]
                for task in as_completed(tasks):
                    offset, value = task.result(); stream.seek(offset); stream.write(value)
            assert partial.stat().st_size == size and hashlib.sha256(partial.read_bytes()).hexdigest() == item['sha256']
            partial.rename(path); item['local'] = str(path.relative_to(ROOT)); item['verified'] = True
            print(json.dumps(dict(package=item['package'], bytes=size, verified=True)), flush=True)
        command = [sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--no-compile',
                   '--target', str(TARGET)]+[str(ROOT/r['local']) for r in records]
        with (out/'install.log').open('x') as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        assert not (TARGET/'torch').exists() and not (TARGET/'numpy').exists()
        write(out/'completed.json', dict(at=datetime.now().astimezone().isoformat(), completed=True, wheels=records,
              target=str(TARGET.relative_to(ROOT)), live_venv_not_modified=True, install_command=command,
              protocol_sha256=hashlib.sha256((BASE/'docs/exploration-r66-perceptual-metric-protocol.md').read_bytes()).hexdigest(),
              scope='Only separate additional-package target; optional EXR/ML checkpoints not installed; no metric score/import/GPU yet.'))
        complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(completed_normally=complete, seconds=time.perf_counter()-start,
              ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__': main()
