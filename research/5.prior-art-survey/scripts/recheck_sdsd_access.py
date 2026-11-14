"""One bounded public access recheck after the previous Drive quota's24h interval."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
import re
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'research/5.prior-art-survey/process/data-readiness-r16'


def main():
    at = datetime.now().astimezone().isoformat()
    old = json.loads((BASE/'drive-download-response.json').read_text())
    age = (datetime.fromisoformat(at)-datetime.fromisoformat(old['time'])).total_seconds()
    assert age > 24*3600
    output = BASE/'access-recheck-20261113.json'; assert not output.exists()
    records = []
    for source in ['indoor_png-listing.json', 'outdoor_png-listing.json']:
        listing = json.loads((BASE/source).read_text())
        item, = [row for row in listing['children'] if row['name'].endswith('.zip')]
        url = 'https://drive.usercontent.google.com/download?id='+item['id']+'&export=download&confirm=t'
        try:
            request = Request(url, headers={'Range': 'bytes=-65536', 'Accept-Encoding': 'identity'})
            with urlopen(request, timeout=30) as response:
                body = response.read(65536)
                headers = {key: response.headers[key] for key in ['Content-Type', 'Content-Length', 'Content-Range', 'ETag', 'Date']
                           if key in response.headers}
                record = dict(name=item['name'], url=url, status=response.status, headers=headers,
                              actual_read_bytes=len(body), response_sha256=hashlib.sha256(body).hexdigest(),
                              prefix_hex=body[:16].hex(), title=re.findall(r'<title>(.*?)</title>', body.decode(errors='replace')),
                              response_kind='HTML' if body.lstrip().startswith(b'<!DOCTYPE html') or b'<html' in body[:256].lower() else 'unclassified')
        except Exception as error:
            record = dict(name=item['name'], url=url, error=repr(error))
        records.append(record)
    output.write_text(json.dumps(dict(at=at, prior_check=old['time'], seconds_since_prior=age,
          maximum_body_bytes_per_endpoint=65536, endpoint_attempts=2, records=records,
          source_listing_sha256={name: hashlib.sha256((BASE/name).read_bytes()).hexdigest()
                                 for name in ['indoor_png-listing.json', 'outdoor_png-listing.json']},
          images_decoded=0, new_enhancement_CNN=0, source='Same author-linked public Drive only; no retry/bypass/mirror.',
          scope='Bounded response recheck only. If Range ignored, read prefix may contain archive payload; no dataset/paired-validity claim.'), indent=2)+'\n')
    print(json.dumps(records))


if __name__ == '__main__': main()
