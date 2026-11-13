"""Read only HEAD headers for the two catalogued official scene1 archives."""
from datetime import datetime
import hashlib
import json
import time
import traceback
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from catalog_crvd import ASSETS, BASE, PROTOCOL, ROOT, mounted_data, write


def main():
    mounted_data()
    source = BASE/'process/external-crvd/catalog003/manifest.json'
    manifest = json.loads(source.read_text())
    assert manifest['completed'] and len(manifest['selected_training_archives']) == 2
    out = BASE/'process/external-crvd/head001'
    out.mkdir(exist_ok=False)
    write(out/'config.json', dict(at=datetime.now().astimezone().isoformat(),
          catalog_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
          protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
          method='HEAD', max_requests=2, no_response_body_read=True, no_retries=True))
    started, completed, requests, rows = time.perf_counter(), False, 0, []
    try:
        for entry in manifest['selected_training_archives']:
            assert entry['name'] == entry['role']+'_scene1.zip'
            url = 'https://drive.usercontent.google.com/download?id='+entry['id']+'&export=download&confirm=t'
            requests += 1
            request = Request(url, method='HEAD', headers={'Accept-Encoding': 'identity'})
            with urlopen(request, timeout=30) as response:
                row = dict(entry, status=response.status, host=urlparse(response.url).hostname,
                           content_type=response.headers.get_content_type(),
                           content_length=response.headers.get('Content-Length'),
                           disposition=response.headers.get('Content-Disposition'),
                           accept_ranges=response.headers.get('Accept-Ranges'),
                           etag=response.headers.get('ETag'), body_bytes_read=0)
                rows.append(row)
                write(out/(entry['role']+'.json'), row)
                assert row['status'] == 200 and row['content_type'] in ['application/zip', 'application/octet-stream']
                assert row['content_length'] is not None and int(row['content_length']) > 0
            print(json.dumps(row), flush=True)
        write(out/'manifest.json', dict(completed=True, archives=rows, body_bytes=0))
        completed = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(out/'runtime.json', dict(completed_normally=completed, requests=requests,
              received_archive_headers=len(rows), body_bytes_read=0, image_decodes=0,
              CNN=0, SGD=0, GPU=0, seconds=time.perf_counter()-started,
              ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__':
    main()
