"""Render a completed internal manuscript for explicit human layout review."""
import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path

from PIL import Image, ImageDraw
from pypdf import PdfReader
import pypdfium2

from project_assets import ASSETS, REPO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=int, required=True)
    args = parser.parse_args()
    paper = REPO/'paper'
    record = json.loads((paper/f'process/build-{args.build:03d}.json').read_text())
    assert record['completed_normally']
    pdf = paper/'manuscript.pdf'
    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
    assert digest == record['pdf_sha256'], 'Review the identified build only'
    assert Path(pypdfium2.__file__).resolve().is_relative_to((ASSETS/'.venv').resolve())
    out = ASSETS/f'cache/paper-build/review-{args.build:03d}'
    out.mkdir(parents=True, exist_ok=False)
    document = pypdfium2.PdfDocument(str(pdf))
    texts = PdfReader(pdf).pages
    assert len(texts) == len(document)
    pages, thumbnails = [], []
    for i, page in enumerate(document):
        bitmap = page.render(scale=1.4)
        rendered = bitmap.to_pil().convert('RGB')
        path = out/f'page-{i+1:02d}.png'
        rendered.save(path)
        thumbnail = rendered.copy()
        thumbnail.thumbnail((390, 555))
        thumbnails.append(thumbnail)
        content = texts[i].extract_text()
        (out/f'page-{i+1:02d}.txt').write_text(content)
        pages.append({'page': i+1, 'render': str(path), 'text_chars': len(content),
                      'unresolved_question_marks': content.count('??')})
        page.close()
    document.close()
    sheet = Image.new('RGB', (800, 585*((len(pages)+1)//2)), '#dddddd')
    draw = ImageDraw.Draw(sheet)
    for i, thumbnail in enumerate(thumbnails):
        x, y = 400*(i % 2), 585*(i//2)
        sheet.paste(thumbnail, (x, y+22))
        draw.text((x+8, y+4), f'Page {i+1}', fill='black')
    sheet.save(out/'contact-sheet.png')
    result = {'created_at': datetime.now().astimezone().isoformat(),
              'pdf_sha256': digest, 'build': args.build, 'pages': pages,
              'renderer_version': importlib.metadata.version('pypdfium2'),
              'renderer_path': str(Path(pypdfium2.__file__).resolve()),
              'contact_sheet': str(out/'contact-sheet.png'),
              'scope': 'Rendering and text extraction only; visual inspection must be recorded separately.'}
    target = paper/f'process/review-{args.build:03d}.json'
    assert not target.exists()
    target.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'pages': len(pages), 'contact_sheet': result['contact_sheet']}))


if __name__ == '__main__':
    main()
