"""Extract/render an active resource's archived PDF; this does not claim reading or Gate approval."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps
from pypdf import PdfReader
import pypdfium2 as pdfium

B = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('resource_id')
    ap.add_argument('archive_path', help='PDF relative to workstation root')
    ap.add_argument('--label', default='paper')
    a = ap.parse_args()
    events = [json.loads(s) for s in (B / 'catalog/analysis-log.jsonl').read_text().splitlines()]
    assert events[-1]['event'] == 'begin' and events[-1]['resource_id'] == a.resource_id
    original = B / a.archive_path
    resource = B / 'resources' / a.resource_id
    target = resource / 'derived/docs' / (a.label + '-assets.json')
    assert not target.exists(), 'Preserve earlier derived assets/version manifest'
    cache = B / '../../.cache/relatedness-survey/derived' / a.resource_id
    files = []

    def write_link(relative, raw):
        dest = cache / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        link = resource / 'derived' / relative
        link.parent.mkdir(parents=True, exist_ok=True)
        assert not link.exists() and not link.is_symlink(), link
        link.symlink_to(os.path.relpath(dest.resolve(), link.parent))
        files.append({'path':str(link.relative_to(resource)),
                      'sha256':hashlib.sha256(raw).hexdigest()})

    reader = PdfReader(original)
    text = '\n\n'.join(f'\n=== PHYSICAL PDF PAGE {i+1} ===\n' + (p.extract_text() or '')
                       for i, p in enumerate(reader.pages))
    write_link('docs/' + a.label + '-text.txt', text.encode())
    doc = pdfium.PdfDocument(original)
    columns = 3
    contact = Image.new('RGB', (columns*490, ((len(doc)+columns-1)//columns)*670), '#eeeeee')
    draw = ImageDraw.Draw(contact)
    import io
    for i in range(len(doc)):
        page = doc[i]
        im = page.render(scale=1.6).to_pil().convert('RGB')
        data = io.BytesIO()
        im.save(data, format='PNG')
        write_link(f'figures/{a.label}-p{i+1:02}.png', data.getvalue())
        thumb = ImageOps.contain(im, (470, 635))
        x, y = (i % columns)*490+10, (i//columns)*670
        draw.text((x, y+4), f'{a.label} physical page {i+1}', fill='black')
        contact.paste(thumb, (x, y+25))
        page.close()
    data = io.BytesIO()
    contact.save(data, format='PNG')
    write_link(f'figures/{a.label}-contact.png', data.getvalue())
    doc.close()
    target.write_text(json.dumps({'resource_id':a.resource_id, 'original':a.archive_path,
                                 'original_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),
                                 'pages':len(reader.pages), 'files':files,
                                 'reading_or_visual_review_claim':False,
                                 'method':'pypdf text; pypdfium2 1.6x rendering; private cached assets'},
                                ensure_ascii=False, indent=2)+'\n')
    print('Prepared', a.resource_id, len(reader.pages), 'pages; no reading/approval claim')


if __name__ == '__main__':
    main()
