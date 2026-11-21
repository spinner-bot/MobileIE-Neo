"""Gate1 identity scans of registered author PDFs; never run research code.

This checks actual terminal HTTP receipts, SHA256, and immutable Git blob IDs.
It extracts only identity excerpts and normalized reference-identity matches.
The output is not a claim of scientific full reading, OCR, or Gate2 approval.
"""
import hashlib
import json
import re
from pathlib import Path

from pypdf import PdfReader

B = Path(__file__).resolve().parents[3]
A = Path(__file__).resolve().parent
OUTPUT = A / 'author-pdf-identity'
NEEDLES = ['mobileie', '250701838', 'iccv51701202502038',
           'anextremelylightweightandeffectiveconvnet', 'hailongyan',
           'litedenoisenet']


def main():
    OUTPUT.mkdir(exist_ok=True)
    jobs = json.loads((A / 'author-pdf-jobs008.json').read_text())
    finished = []
    pending = []
    for job in jobs:
        receipt_path = B / 'collection/receipts' / (job['label'] + '.json')
        if not receipt_path.exists():
            pending.append(job['label'])
            continue
        receipt = json.loads(receipt_path.read_text())
        if not receipt.get('complete') or receipt.get('status') != 'success':
            pending.append(job['label'])
            continue
        original = B / receipt['archive_path']
        data = original.read_bytes()
        assert len(data) == receipt['bytes'] == job['expected_bytes']
        assert hashlib.sha256(data).hexdigest() == receipt['sha256']
        blob = b'blob ' + str(len(data)).encode() + b'\0' + data
        assert hashlib.sha1(blob).hexdigest() == job['git_blob_sha1']
        path = OUTPUT / (job['label'] + '.json')
        if path.exists():
            previous = json.loads(path.read_text())
            assert previous['source_sha256'] == receipt['sha256']
            assert previous['job'] == job and previous['git_blob_verified']
            finished.append(previous)
            continue
        reader = PdfReader(original)
        pages, hits, first = [], [], ''
        for number, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ''
            normalized = re.sub('[^a-z0-9]', '', text.lower())
            pages.append(dict(page=number, extracted_characters=len(text)))
            if number == 1:
                first = text[:1600]
            hits.extend(dict(page=number, needle=needle)
                        for needle in NEEDLES if needle in normalized)
        record = dict(
            scope='Gate1 identity/reference scan only, no scientific full reading/Gate2',
            job=job, source_receipt=str(receipt_path.relative_to(B)),
            source_sha256=receipt['sha256'], git_blob_verified=True,
            pdf_metadata_title=str((reader.metadata or {}).get('/Title', '')),
            first_page_identity_excerpt=first, pages=pages, identity_hits=hits)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
        finished.append(record)
    print(json.dumps(dict(total=len(jobs), finished=len(finished), pending=pending,
                          matching_documents=[r['job']['label'] for r in finished
                                              if r['identity_hits']],
                          documents=[dict(label=r['job']['label'], pages=len(r['pages']),
                                          title=r['pdf_metadata_title'][:100])
                                     for r in finished]), ensure_ascii=False))


if __name__ == '__main__':
    main()
