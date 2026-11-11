"""Recover slow public PyPI downloads with four bounded, hash-verified range requests."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen

from project_assets import ASSETS, mounted_data, stamp, write_json


def download_part(url, start, end, total):
    for attempt in range(4):
        try:
            request = Request(url, headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'})
            with urlopen(request, timeout=30) as response:
                if total > end-start+1:
                    assert response.status == 206
                    assert response.headers['Content-Range'] == f'bytes {start}-{end}/{total}'
                value = response.read()
            assert len(value) == end-start+1
            return start, value
        except Exception:
            if attempt == 3:
                raise
            time.sleep(attempt+1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install-log', required=True)
    parser.add_argument('--record', required=True)
    args = parser.parse_args()
    mounted_data()
    folder = ASSETS/'wheelhouse'
    folder.mkdir(parents=True, exist_ok=True)
    filenames = sorted(set(re.findall(r'Downloading ([A-Za-z0-9_.+\-]+\.whl)\.metadata', Path(args.install_log).read_text())))
    assert len(filenames) >= 20, 'require the resolved research dependency closure, not a partial resolver log'
    records = []
    for filename in filenames:
        package, version = filename.split('-')[:2]
        metadata_url = f'https://pypi.org/pypi/{package}/{version}/json'
        with urlopen(metadata_url, timeout=30) as response:
            metadata = json.load(response)
        item = next(f for f in metadata['urls'] if f['filename'] == filename)
        url, size, expected = item['url'], item['size'], item['digests']['sha256']
        assert url.startswith('https://files.pythonhosted.org/')
        target = folder/filename
        if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            partial = target.with_suffix('.whl.partial')
            chunk = 256*1024
            start_time = time.monotonic()
            with partial.open('wb') as stream, ThreadPoolExecutor(max_workers=4) as pool:
                stream.truncate(size)
                tasks = [pool.submit(download_part, url, s, min(s+chunk, size)-1, size) for s in range(0, size, chunk)]
                for task in as_completed(tasks):
                    offset, value = task.result()
                    stream.seek(offset)
                    stream.write(value)
            assert partial.stat().st_size == size
            assert hashlib.sha256(partial.read_bytes()).hexdigest() == expected
            partial.rename(target)
            print(json.dumps({'wheel': filename, 'bytes': size, 'seconds': time.monotonic()-start_time, 'verified': True}), flush=True)
        records.append({'filename': filename, 'metadata_url': metadata_url, 'download_url': url, 'bytes': size,
                        'official_sha256': expected, 'actual_sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
        write_json(args.record, {'observed_at': stamp(), 'complete': False, 'source': 'official PyPI JSON and files.pythonhosted.org only',
                                'concurrency': 4, 'wheels': records})
    write_json(args.record, {'observed_at': stamp(), 'complete': True, 'source': 'official PyPI JSON and files.pythonhosted.org only',
                            'concurrency': 4, 'wheels': records})


if __name__ == '__main__':
    main()
