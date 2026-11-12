"""Acquire the author's public sample RAR via its normal anonymous download form."""
import hashlib
import html.parser
import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime

from dynamic_training_r34 import BASE, ROOT, digest

FILE_ID = '1bIaPaFeOWDwuhWgLXraY1FrSZBAqJF2T'
EXPECTED = 1170848021
PUBLIC_URL = 'https://drive.google.com/uc?export=download&id='+FILE_ID


class DownloadForm(html.parser.HTMLParser):
    def __init__(self):
        super().__init__(); self.action, self.values = None, {}

    def handle_starttag(self, tag, attributes):
        values = dict(attributes)
        if tag == 'form': self.action = values.get('action')
        if tag == 'input' and values.get('name'):
            self.values[values['name']] = values.get('value', '')


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    out = BASE/'process/external-night-wenzhou/acquisition001'; out.mkdir(parents=True, exist_ok=False)
    cache = ROOT/'.cache/datasets/Night-Wenzhou/author-samples/attempt001'
    assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    cache.mkdir(parents=True, exist_ok=False)
    partial, final = cache/'Wenzhou-Night.rar.part', cache/'Wenzhou-Night.rar'
    start, complete, count = time.perf_counter(), False, 0
    try:
        with urllib.request.urlopen(PUBLIC_URL, timeout=30) as response:
            value = response.read(65536)
            assert len(value) < 65536 and response.headers.get_content_type() == 'text/html'
        form = DownloadForm(); form.feed(value.decode('utf8'))
        assert form.action == 'https://drive.usercontent.google.com/download'
        assert form.values['id'] == FILE_ID and form.values.get('confirm') == 't'
        # Public one-time form fields stay in process memory, never in logs/manifests.
        private_request_url = form.action+'?'+urllib.parse.urlencode(form.values)
        checksum = hashlib.sha256(); last_progress = 0
        with urllib.request.urlopen(private_request_url, timeout=30) as response:
            assert response.status == 200 and response.headers.get_content_type() == 'application/octet-stream'
            assert int(response.headers['Content-Length']) == EXPECTED
            write(out/'source.json', {'at': datetime.now().astimezone().isoformat(), 'public_url': PUBLIC_URL,
                  'source_issue': 'https://github.com/ShenZheng2000/LLIE_Survey/issues/6#issuecomment-2597100240',
                  'file_id': FILE_ID, 'bytes_expected': EXPECTED, 'normal_anonymous_confirmation': True,
                  'Content-Disposition': response.headers.get('Content-Disposition'),
                  'temporary_request_fields_logged': False,
                  'protocol_sha256': digest(BASE/'docs/exploration-r52-wenzhou-samples-protocol.md')})
            with partial.open('xb') as stream:
                while block := response.read(1024*1024):
                    if count == 0: assert block.startswith((b'Rar!\x1a\x07\x00', b'Rar!\x1a\x07\x01\x00'))
                    stream.write(block); checksum.update(block); count += len(block)
                    assert count <= EXPECTED
                    if count-last_progress >= 64*1024*1024:
                        last_progress = count
                        print(json.dumps({'downloaded_bytes': count, 'expected': EXPECTED}), flush=True)
        assert count == EXPECTED
        partial.rename(final)
        write(out/'manifest.json', {'at': datetime.now().astimezone().isoformat(), 'path': str(final.relative_to(ROOT)),
              'bytes': count, 'sha256': checksum.hexdigest(), 'checksum_is_author_provided': False,
              'download_complete': True, 'archive_members_extracted': 0, 'video_frames_decoded': 0,
              'new_CNN': 0, 'scope': 'Author replacement sample archive, not full Night Wenzhou or completed night evaluation.'})
        complete = True
    except BaseException as error:
        # Exception URLs can carry normal form tokens; retain diagnosis without them.
        write(out/'failure.json', {'type': type(error).__name__,
              'message': re.sub(r'https?://\S+', '<request-url-redacted>', str(error)),
              'downloaded_bytes': count, 'partial_preserved': partial.exists()})
        raise RuntimeError('Public sample acquisition failed; sanitized diagnosis retained') from None
    finally:
        write(out/'runtime.json', {'completed_normally': complete, 'downloaded_bytes': count,
              'seconds': time.perf_counter()-start, 'ended': datetime.now().astimezone().isoformat()})


if __name__ == '__main__': main()
