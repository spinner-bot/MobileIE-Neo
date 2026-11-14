"""Read anonymous author-linked DID share metadata, without binary downloads."""
from datetime import datetime
from http.cookiejar import CookieJar
import json
import re
from urllib.parse import urlencode
from urllib.request import build_opener, HTTPCookieProcessor, Request

from dynamic_training_r34 import BASE

URL = 'https://pan.baidu.com/s/1dSx-XxKd7MVwh9xPLvZn3A?pwd=nomy'


def main():
    out = BASE/'process/external-did/public-directory-metadata.json'
    assert not out.exists()
    out.parent.mkdir(parents=True, exist_ok=True)
    client = build_opener(HTTPCookieProcessor(CookieJar()))
    client.addheaders = [('User-Agent', 'Mozilla/5.0')]
    record = {'observed_at': datetime.now().astimezone().isoformat(), 'author_share_url': URL,
              'scope': 'Anonymous author-provided public code, metadata only; no login, captcha bypass, binary downloads, saved cookies or token logging.'}
    try:
        client.open(URL, timeout=20).close()
        request = Request('https://pan.baidu.com/share/verify?surl=dSx-XxKd7MVwh9xPLvZn3A',
                          data=urlencode({'pwd': 'nomy'}).encode(), headers={'Referer': URL})
        with client.open(request, timeout=20) as response:
            verify = json.load(response)
        record['public_code_errno'] = verify.get('errno')
        assert verify.get('errno') == 0, 'Public code not accepted; stop without retries'
        with client.open(URL, timeout=20) as response:
            text = response.read(1048576).decode('utf8', 'replace')
        def number(name):
            match = re.search(r'["\']'+name+r'["\']\s*:\s*["\']?(\d+)', text)
            assert match, 'Required public share ID absent: '+name
            return match.group(1)
        shareid, uk = number('shareid'), number('share_uk')
        # The full JS object is not JSON. Only decode its JSON-valued file_list.
        match = re.search(r'"file_list"\s*:\s*([\[{])', text)
        assert match, 'No parseable file metadata; do not evaluate page JavaScript'
        files, _ = json.JSONDecoder().raw_decode(text[match.start(1):])
        if isinstance(files, dict):
            files = files.get('list', [])
        safe = lambda row: {k: row[k] for k in ['server_filename', 'path', 'isdir', 'size', 'fs_id', 'md5'] if k in row}
        record['root_metadata'] = [safe(row) for row in files]
        assert len(files) == 1 and files[0]['server_filename'] == 'DID' and files[0]['isdir'] == 1
        directories = [files[0]['path']]
        record['directory_pages'] = []
        # Three metadata requests are enough to establish basic layout; no broad crawl.
        for iteration in range(3):
            if not directories:
                break
            directory = directories.pop(0)
            query = urlencode({'shareid': shareid, 'uk': uk, 'dir': directory, 'order': 'name',
                               'desc': 0, 'showempty': 0, 'web': 1, 'page': 1, 'num': 100})
            with client.open(Request('https://pan.baidu.com/share/list?'+query, headers={'Referer': URL}), timeout=20) as response:
                value = json.load(response)
            page = {'directory': directory, 'errno': value.get('errno')}
            record['directory_pages'].append(page)
            if value.get('errno') != 0:
                break
            rows = value.get('list', [])
            page['files'] = [safe(row) for row in rows]
            page['pagination_may_be_incomplete'] = len(rows) >= 100
            directories.extend(row['path'] for row in rows if row.get('isdir'))
        record['directory_metadata_obtained'] = any(page.get('files') for page in record['directory_pages'])
        record['binary_download_attempted'] = False
    except Exception as error:
        # Exception messages from HTTP may contain sensitive request query values;
        # keep only the type and safe assertions, never session/token contents.
        record['error_type'] = type(error).__name__
        if isinstance(error, AssertionError):
            record['assertion'] = str(error)
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
