"""Prepare a required appended publication review without touching the closed original audit."""
import argparse
import difflib
import hashlib
import io
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps
from pypdf import PdfReader
import pypdfium2 as pdfium

from catalog_state import B, load_catalog, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('resource_id')
    parser.add_argument('review_number')
    parser.add_argument('receipt')
    parser.add_argument('prior_pdf')
    args = parser.parse_args()
    load_catalog(require_advancement=False)
    relative = f'resources/{args.resource_id}/derived/reviews/{args.review_number}'
    requirements = []
    for p in (B / 'collection/gate1-addenda').glob('*/status.json'):
        status = json.loads(p.read_text())
        if status.get('gate1_passed'):
            requirements.extend(status.get('required_reviews', []))
    assert any(r['review_dir'] == relative for r in requirements)
    directory = B / relative
    asset_manifest = directory / 'docs/paper-assets.json'
    assert not asset_manifest.exists(), 'Preserve prior prepared review/version'
    receipt = json.loads((B / args.receipt).read_text())
    original = B / receipt['archive_path']
    assert receipt.get('complete') and receipt['http_status'] == 200
    assert sha(original) == receipt['sha256']
    resource = B / 'resources' / args.resource_id
    assert (resource / 'derived/report/integrity.json').is_file()
    cache = (B / '../../.cache/relatedness-survey/derived' / args.resource_id /
             'reviews' / args.review_number).resolve()
    files = []

    def link_file(relative_path, data):
        destination = cache / relative_path
        target = directory / relative_path
        assert not destination.exists() and not target.exists() and not target.is_symlink()
        destination.parent.mkdir(parents=True, exist_ok=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        target.symlink_to(os.path.relpath(destination, target.parent))
        files.append({'path':relative_path, 'sha256':hashlib.sha256(data).hexdigest()})

    reader = PdfReader(original)
    pages = [p.extract_text() or '' for p in reader.pages]
    prior = PdfReader(B / args.prior_pdf)
    old_pages = [p.extract_text() or '' for p in prior.pages]
    assert len(pages) == len(old_pages)
    text = '\n\n'.join(f'=== PHYSICAL PDF PAGE {i+1} ===\n{t}' for i,t in enumerate(pages))
    link_file('docs/paper-text.txt', text.encode())
    comparison = []
    for i, (old, new) in enumerate(zip(old_pages, pages), 1):
        comparison.append({'page':i, 'extracted_text_equal':old == new,
                           'diff':'\n'.join(difflib.unified_diff(old.splitlines(), new.splitlines(),
                                                               fromfile=f'arxiv-v1-p{i}',
                                                               tofile=f'pmlr-p{i}'))})
    document = pdfium.PdfDocument(original)
    contact = Image.new('RGB', (1470, ((len(document)+2)//3)*670), '#eeeeee')
    draw = ImageDraw.Draw(contact)
    rendered = []
    for i in range(len(document)):
        page = document[i]
        picture = page.render(scale=1.6).to_pil().convert('RGB')
        output = io.BytesIO()
        picture.save(output, format='PNG')
        data = output.getvalue()
        link_file(f'figures/paper-p{i+1:02}.png', data)
        prior_image = resource / f'derived/figures/paper-p{i+1:02}.png'
        rendered.append({'page':i+1, 'rendered_png_equal_to_original_preprint':
                         prior_image.is_file() and sha(prior_image) == hashlib.sha256(data).hexdigest()})
        thumb = ImageOps.contain(picture, (470,635))
        x,y = (i%3)*490+10,(i//3)*670
        draw.text((x,y+4), f'PMLR physical page {i+1}', fill='black')
        contact.paste(thumb,(x,y+25))
        page.close()
    output = io.BytesIO()
    contact.save(output, format='PNG')
    link_file('figures/paper-contact.png', output.getvalue())
    document.close()
    (directory / 'docs/version-compare.json').write_text(json.dumps({
        'source':receipt['archive_path'], 'sha256':receipt['sha256'],
        'prior_source':args.prior_pdf, 'prior_sha256':sha(B / args.prior_pdf),
        'pages':comparison, 'rendered_comparison':rendered,
        'scope':'Document version/text/PNG comparison only; no scientific model evaluation or reading claim',
    }, ensure_ascii=False, indent=2) + '\n')
    asset_manifest.write_text(json.dumps({
        'resource_id':args.resource_id, 'review_id':args.review_number,
        'original':receipt['archive_path'], 'original_sha256':receipt['sha256'],
        'pages':len(pages), 'files':files, 'reading_or_visual_review_claim':False,
        'method':'pypdf full text, pypdfium2 1.6x pages and contact sheet in owned cache',
    }, ensure_ascii=False, indent=2) + '\n')
    print('Prepared appended review', relative, len(pages), 'pages; original audit unchanged.')


if __name__ == '__main__':
    main()
