"""One ordinary public Range recovery, preserving the original premature EOF."""
import hashlib
import json
import re
import shutil
import time
import urllib.parse
import urllib.request
from datetime import datetime

from dynamic_training_r34 import BASE, ROOT, digest
from fetch_wenzhou_samples_r52 import DownloadForm, EXPECTED, FILE_ID, PUBLIC_URL, write


def main():
    process = BASE/'process/external-night-wenzhou'
    old = json.loads((process/'acquisition001/runtime.json').read_text())
    assert old['completed_normally'] is False
    assert old['downloaded_bytes'] == 1148839705
    original = ROOT/'.cache/datasets/Night-Wenzhou/author-samples/attempt001/Wenzhou-Night.rar.part'
    cache = original.parent.parent/'attempt002'
    out = process/'acquisition002'
    assert original.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    assert original.stat().st_size == old['downloaded_bytes']
    out.mkdir(exist_ok=False); cache.mkdir(exist_ok=False)
    partial, final = cache/'Wenzhou-Night.rar.part', cache/'Wenzhou-Night.rar'
    offset, overlap = original.stat().st_size, 1024*1024
    start, complete, count = time.perf_counter(), False, offset
    original_sha = digest(original)
    try:
        write(out/'config.json', dict(at=datetime.now().astimezone().isoformat(),
              original=str(original.relative_to(ROOT)), original_bytes=offset, original_sha256=original_sha,
              range_start=offset-overlap, range_end=EXPECTED-1, overlap_bytes=overlap,
              technical_retries=0, protocol_sha256=digest(BASE/'docs/exploration-r52-wenzhou-range-recovery-protocol.md'),
              script_sha256=digest(__file__), old_runtime_sha256=digest(process/'acquisition001/runtime.json')))
        shutil.copyfile(original, partial)
        assert digest(partial) == original_sha
        with urllib.request.urlopen(PUBLIC_URL, timeout=30) as response:
            value = response.read(65536)
            assert len(value) < 65536 and response.headers.get_content_type() == 'text/html'
        form = DownloadForm(); form.feed(value.decode('utf8'))
        assert form.action == 'https://drive.usercontent.google.com/download'
        assert form.values['id'] == FILE_ID and form.values.get('confirm') == 't'
        private_url = form.action+'?'+urllib.parse.urlencode(form.values)
        request = urllib.request.Request(private_url, headers={
            'Range': f'bytes={offset-overlap}-{EXPECTED-1}', 'Accept-Encoding': 'identity'})
        with urllib.request.urlopen(request, timeout=30) as response:
            headers = response.headers
            source = dict(at=datetime.now().astimezone().isoformat(), public_url=PUBLIC_URL,
                  status=response.status, content_range=headers.get('Content-Range'),
                  content_length=headers.get('Content-Length'), content_type=headers.get_content_type(),
                  content_disposition=headers.get('Content-Disposition'), etag=headers.get('ETag'),
                  last_modified=headers.get('Last-Modified'), temporary_request_fields_logged=False)
            write(out/'source.json', source)
            assert response.status == 206, 'Server did not honor Range; no full-body fallback.'
            assert headers.get('Content-Range') == f'bytes {offset-overlap}-{EXPECTED-1}/{EXPECTED}'
            assert int(headers['Content-Length']) == EXPECTED-offset+overlap
            assert headers.get_content_type() == 'application/octet-stream'
            prior = json.loads((process/'acquisition001/source.json').read_text())
            assert headers.get('Content-Disposition') == prior['Content-Disposition']
            with original.open('rb') as stream:
                stream.seek(offset-overlap); previous = stream.read(overlap)
            received = response.read(overlap)
            assert len(received) == overlap and received == previous, 'Source overlap mismatch.'
            write(out/'overlap.json', dict(bytes=overlap, equal=True,
                  sha256=hashlib.sha256(received).hexdigest(), first_new_byte=offset))
            with partial.open('ab') as stream:
                while block := response.read(1024*1024):
                    stream.write(block); count += len(block)
                    assert count <= EXPECTED
                    print(json.dumps(dict(total_bytes=count, expected=EXPECTED)), flush=True)
        assert count == EXPECTED and partial.stat().st_size == EXPECTED
        with partial.open('rb') as stream:
            assert stream.read(8).startswith((b'Rar!\x1a\x07\x00', b'Rar!\x1a\x07\x01\x00'))
        assert original.stat().st_size == offset and digest(original) == original_sha
        full_sha = digest(partial); partial.rename(final)
        write(out/'manifest.json', dict(at=datetime.now().astimezone().isoformat(),
              path=str(final.relative_to(ROOT)), bytes=count, sha256=full_sha,
              checksum_is_author_provided=False, download_complete=True, original_partial_preserved=True,
              original_partial_sha256=original_sha, new_payload_bytes=count-offset,
              overlap_bytes=overlap, overlap_equal=True, technical_retries=0,
              archive_members_extracted=0, video_frames_decoded=0, new_CNN=0,
              scope='Author partial sample backup, recovered from EOF; not full benchmark or night evaluation.'))
        complete = True
    except BaseException as error:
        write(out/'failure.json', dict(type=type(error).__name__,
              message=re.sub(r'https?://\S+', '<request-url-redacted>', str(error)),
              total_bytes=count, original_partial_preserved=original.exists(), recovered_partial=partial.exists()))
        raise RuntimeError('One public Range recovery failed; sanitized diagnosis retained; no retry.') from None
    finally:
        write(out/'runtime.json', dict(completed_normally=complete, total_bytes=count,
              new_payload_bytes=count-offset, seconds=time.perf_counter()-start,
              ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__':
    main()
