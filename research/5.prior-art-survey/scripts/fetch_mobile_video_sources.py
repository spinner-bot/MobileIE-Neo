"""Archive two public author PDFs in owned data; no dataset/model download."""
import hashlib
import json
import urllib.request
from datetime import datetime
from pathlib import Path

import pypdfium2

ROOT = Path(__file__).resolve().parents[3]
SOURCES = {
    'raw2raw-cvpr2026': 'https://lumen-cvpr26.github.io/raw-to-raw-extreme-low-light-enhancement/assets/pdfs/paper.pdf',
    'ubienlight-imwut2026': 'https://zhouzimu.github.io/paper/ubicomp26-wang.pdf',
}


def main():
    cache = ROOT/'.cache/papers/mobile-video-2026/attempt001'
    assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    cache.mkdir(parents=True, exist_ok=False)
    out = ROOT/'research/5.prior-art-survey/process/mobile-video-2026'
    out.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        partial = cache/f'{name}.pdf.part'
        with urllib.request.urlopen(url, timeout=30) as response, partial.open('xb') as stream:
            expected = response.headers.get('Content-Length')
            while block := response.read(256*1024): stream.write(block)
        value = partial.read_bytes()
        assert value.startswith(b'%PDF') and b'%%EOF' in value[-1024:]
        assert expected is None or len(value) == int(expected)
        path = partial.with_suffix(''); partial.rename(path)
        pdf = pypdfium2.PdfDocument(path)
        text_path = cache/f'{name}.txt'
        with text_path.open('x') as stream:
            for index in range(len(pdf)):
                page = pdf[index]; textpage = page.get_textpage()
                stream.write(f'\n--- PAGE {index+1} ---\n'+textpage.get_text_range()+'\n')
                textpage.close(); page.close()
        record = {'at': datetime.now().astimezone().isoformat(), 'url': url,
                  'path': str(path.relative_to(ROOT)), 'bytes': len(value),
                  'sha256': hashlib.sha256(value).hexdigest(), 'pages': len(pdf),
                  'text_path': str(text_path.relative_to(ROOT)),
                  'scope': 'Public author paper only; acquisition is not full reading or model reproduction.'}
        pdf.close()
        (out/f'{name}-acquisition.json').write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps(record), flush=True)


if __name__ == '__main__': main()
